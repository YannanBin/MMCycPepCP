import os
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem
import multiprocessing as mp
import logging
import signal
from contextlib import contextmanager
import re

from rdkit.Chem.rdDistGeom import ETKDGv3, EmbedMolecule
from rdkit.Chem.rdForceFieldHelpers import MMFFOptimizeMolecule, UFFOptimizeMolecule

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# Timeout context manager
class TimeoutException(Exception):
    pass


@contextmanager
def timeout(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("处理超时")

    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)


def clean_filename(s):
    """
    Clean string to make it safe for filenames.

    Args:
        s: Input string.

    Returns:
        Cleaned string.
    """
    # Replace invalid characters with underscore
    return re.sub(r'[^\w\-]', '_', s.strip())


def load_molecule(input_mol):
    """
    Load molecule from input string (SMILES, InChI, PDB, MOL).

    Args:
        input_mol: Input string (SMILES, InChI, PDB block, MOL block).

    Returns:
        mol_type: Type of input ('SMILES', 'INCHI', 'PDB', 'MOL').
        mol: RDKit molecule object or None if invalid.
    """
    if 'InChI' in input_mol:
        method = Chem.MolFromInchi
        mol_type = 'INCHI'
    elif '\n' in input_mol:
        if 'ATOM' in input_mol.upper() or 'HETATM' in input_mol.upper() or 'CONECT' in input_mol.upper():
            method = Chem.MolFromPDBBlock
            mol_type = 'PDB'
        else:
            method = Chem.MolFromMolBlock
            mol_type = 'MOL'
    else:
        method = Chem.MolFromSmiles
        mol_type = 'SMILES'
    try:
        mol = method(input_mol)
    except Exception as e:
        logger.error(f"加载分子失败: {input_mol[:50]}... 错误: {str(e)}")
        mol = None
    if mol_type == 'PDB' and mol is not None:
        mol.RemoveAllConformers()
    return mol_type, mol

"""
def predict_3d_conformation(mol):
    # Generate 3D conformation with RDKit ETKDGv3 embedding and UFF/MMFF optimization.
    # 
    # Args:
    #     mol: RDKit molecule object.
    # 
    # Returns:
    #     mol_3d: Molecule with 3D conformation or None if failed.
    try:
        # Add hydrogens and embed with ETKDGv3
        mol_3d = Chem.AddHs(mol.__copy__())
        etkdg = AllChem.ETKDGv3()
        etkdg.randomSeed = 42
        etkdg.maxIterations = 2000
        etkdg.useRandomCoords = True
        etkdg.numThreads = 0

        # Try multiple conformations
        AllChem.EmbedMultipleConfs(mol_3d, numConfs=5, params=etkdg)
        if mol_3d.GetNumConformers() == 0:
            logger.warning("所有嵌入尝试失败，尝试单次嵌入")
            if AllChem.EmbedMolecule(mol_3d, etkdg) != 0:
                logger.error("单次嵌入失败")
                return None

        # Select lowest energy conformation
        best_conf_id = 0
        best_energy = float('inf')
        for conf_id in range(mol_3d.GetNumConformers()):
            try:
                AllChem.UFFOptimizeMolecule(mol_3d, maxIters=500, confId=conf_id)
                ff = AllChem.UFFGetMoleculeForceField(mol_3d, confId=conf_id)
                energy = ff.CalcEnergy()
                if energy < best_energy:
                    best_energy = energy
                    best_conf_id = conf_id
            except Exception as e:
                logger.warning(f"UFF 优化失败 (构象 {conf_id}): {str(e)}")
                continue

        # Final MMFF optimization on best conformation
        try:
            AllChem.MMFFOptimizeMolecule(mol_3d, maxIters=1000, confId=best_conf_id)
            logger.info(f"最佳构象能量: {best_energy:.2f} (构象 {best_conf_id})")
        except Exception as e:
            logger.warning(f"MMFF 优化失败: {str(e)}，使用 UFF 优化结果")

        # Create new molecule with best conformation
        if mol_3d.GetNumConformers() > 0:
            return Chem.Mol(mol_3d, confId=best_conf_id)
        return None
    except Exception as e:
        logger.error(f"3D 构象生成失败: {str(e)}")
        return None
"""
def predict_3d_conformation(mol):
    """Generate 3D conformation with ETKDGv3 embedding and MMFF/UFF optimization."""
    try:
        # Add hydrogens and embed with ETKDGv3
        mol_3d = Chem.AddHs(mol.__copy__())
        etkdg = ETKDGv3()
        etkdg.randomSeed = 42
        etkdg.maxIterations = 200  # Reduced iterations for faster processing
        etkdg.useRandomCoords = True
        etkdg.numThreads = 1  # Avoid nested parallelism

        # Generate single conformation
        if EmbedMolecule(mol_3d, etkdg) != 0:
            logger.error("Conformation embedding failed")
            return None

        # 优先使用MMFF优化
        try:
            MMFFOptimizeMolecule(mol_3d, maxIters=200)
            logger.info("MMFF optimization succeeded")
        except Exception as e:
            logger.warning(f"MMFF optimization failed, trying UFF: {str(e)}")
            # MMFF失败时回退到UFF
            try:
                UFFOptimizeMolecule(mol_3d, maxIters=200)
                logger.info("UFF optimization succeeded")
            except Exception as e:
                logger.error(f"UFF optimization also failed: {str(e)}")
                return None

        return mol_3d if mol_3d.GetNumConformers() > 0 else None
    except Exception as e:
        logger.error(f"3D conformation generation failed: {str(e)}")
        return None


def save_pdb(mol, filename='molecule.pdb'):
    """
    Save molecule as PDB file.

    Args:
        mol: RDKit molecule object.
        filename: Output PDB file path.

    Returns:
        pdb_block: PDB block string or None if failed.
    """
    try:
        pdb_block = Chem.MolToPDBBlock(mol)
        if pdb_block is None:
            raise ValueError("无法将分子转换为 PDB 格式")
        os.makedirs(os.path.dirname(filename), exist_ok=True)
        with open(filename, 'w') as f:
            f.write(pdb_block)
        return pdb_block
    except Exception as e:
        logger.error(f"保存 PDB 文件失败: {filename}, 错误: {str(e)}")
        return None


def process_smiles(args):
    """
    Process a single SMILES string to generate and save PDB file with timeout.

    Args:
        args: Tuple of (index, smiles, cpkb_id, output_dir).

    Returns:
        bool: True if successful, False otherwise.
    """
    idx, smiles, cpkb_id, output_dir = args
    if not cpkb_id or pd.isna(cpkb_id):
        logger.warning(f"CPKB ID 缺失，使用索引命名: index={idx}, SMILES={smiles[:50]}...")
        cpkb_id = f"sample_{idx}"
    cpkb_id = clean_filename(str(cpkb_id))
    pdb_file = os.path.join(output_dir, f"{cpkb_id}.pdb")

    # Skip if PDB file already exists
    if os.path.exists(pdb_file):
        logger.info(f"PDB 文件已存在，跳过: {pdb_file}")
        return True

    try:
        # with timeout(300):  # 5-minute timeout per SMILES
        mol_type, mol = load_molecule(smiles)
        if mol is None or mol_type != 'SMILES':
            logger.error(f"无效 SMILES 或非 SMILES 输入: {smiles[:50]}...")
            return False
        mol_3d = predict_3d_conformation(mol)
        if mol_3d is None:
            logger.error(f"3D 构象生成失败: {smiles[:50]}...")
            return False
        if save_pdb(mol_3d, pdb_file) is not None:
            return True
        else:
            logger.error(f"PDB 保存失败: {smiles[:50]}...")
            return False
    except TimeoutException:
        logger.error(f"处理超时: {smiles[:50]}... (CPKB ID: {cpkb_id})")
        return False
    except Exception as e:
        logger.error(f"处理 SMILES 失败: {smiles[:50]}... (CPKB ID: {cpkb_id}) 错误: {str(e)}")
        return False


def generate_all_pdb(csv_file, output_dir='./pdb_files', num_processes=None):
    """
    Generate 3D conformations for all SMILES in the CSV file and save as PDB files named by CPKB ID.

    Args:
        csv_file: Path to CSV file containing SMILES and CPKB ID.
        output_dir: Directory to save PDB files.
        num_processes: Number of processes for parallel processing (default: CPU count).

    Returns:
        success_ratio: Ratio of successfully generated PDB files.
    """
    df = pd.read_csv(csv_file)
    if 'SMILES' not in df.columns:
        raise ValueError("CSV 文件必须包含 'SMILES' 列")
    if 'CPKB ID' not in df.columns:
        raise ValueError("CSV 文件必须包含 'CPKB ID' 列")

    # Check for missing or duplicate CPKB IDs
    if df['CPKB ID'].isna().any():
        logger.warning("存在缺失的 CPKB ID，将回退到索引命名")
    if df['CPKB ID'].duplicated().any():
        logger.warning("存在重复的 CPKB ID，可能导致 PDB 文件覆盖")

    smiles_list = df['SMILES'].values
    cpkb_ids = df['CPKB ID'].values
    total_count = len(smiles_list)

    # Set up multiprocessing
    if num_processes is None:
        # num_processes = min(mp.cpu_count(), 8)
        num_processes = mp.cpu_count()
    logger.info(f"使用 {num_processes} 个进程进行并行处理")
    pool = mp.Pool(processes=num_processes)

    # Prepare arguments for parallel processing
    args = [(idx, smiles, cpkb_id, output_dir) for idx, (smiles, cpkb_id) in enumerate(zip(smiles_list, cpkb_ids))]

    # Run parallel processing
    results = pool.map(process_smiles, args)
    pool.close()
    pool.join()

    success_count = sum(results)
    success_ratio = success_count / total_count if total_count > 0 else 0.0
    logger.info(f"PDB 文件生成成功比例: {success_ratio:.4f} ({success_count}/{total_count})")

    # Save failed SMILES for debugging
    failed_smiles = [(smiles, cpkb_id) for idx, (smiles, cpkb_id) in enumerate(zip(smiles_list, cpkb_ids)) if
                     not results[idx]]
    if failed_smiles:
        with open(os.path.join(output_dir, 'failed_smiles.txt'), 'w') as f:
            f.write('\n'.join(f"{smiles}\t{cpkb_id}" for smiles, cpkb_id in failed_smiles))
        logger.info(f"失败的 SMILES 已保存至 {os.path.join(output_dir, 'failed_smiles.txt')}")

    return success_ratio


if __name__ == '__main__':
    csv_file = 'datasets/lastfinal_cyclic_peptides.csv'
    generate_all_pdb(csv_file)