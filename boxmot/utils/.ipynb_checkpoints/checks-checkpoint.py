import subprocess
from pathlib import Path
import logging
from importlib.metadata import distributions, PackageNotFoundError, version
from packaging.requirements import Requirement
from packaging.specifiers import SpecifierSet

# 定义日志（保持原有逻辑）
LOGGER = logging.getLogger(__name__)

# 定义 REQUIREMENTS 路径（根据你的实际路径调整）
REQUIREMENTS = Path(__file__).parent.parent / 'requirements.txt'  # 示例路径，按需调整

class RequirementsChecker:
    
    def check_requirements(self):
        # 安全打开 requirements 文件并解析
        with REQUIREMENTS.open(encoding='utf-8') as f:
            requirements = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            self.check_packages(requirements)  # 直接传字符串列表，内部统一解析

    def check_packages(self, requirements, cmds=''):
        """Test that each required package is available（兼容字符串/Requirement对象输入）"""
        missing_packages = []
        # 获取已安装的所有包（名称: 版本）
        installed = {pkg.name.lower(): version(pkg.name) for pkg in distributions()}
        
        for req in requirements:
            # 统一将输入转为 Requirement 对象（核心修复点）
            if isinstance(req, str):
                try:
                    req_obj = Requirement(req)
                except Exception as e:
                    LOGGER.error(f'Failed to parse requirement "{req}": {e}')
                    missing_packages.append(req)
                    continue
            else:
                req_obj = req  # 如果是已解析的 Requirement 对象，直接使用
            
            pkg_name = req_obj.name.lower()
            spec = SpecifierSet(str(req_obj.specifier)) if req_obj.specifier else SpecifierSet()
            
            try:
                # 检查包是否安装
                if pkg_name not in installed:
                    raise PackageNotFoundError(f"Package {req_obj.name} not found")
                # 检查版本是否符合要求
                installed_ver = installed[pkg_name]
                if installed_ver not in spec:
                    raise ValueError(f"Package {req_obj.name} version {installed_ver} does not match {spec}")
            except (PackageNotFoundError, ValueError) as e:
                LOGGER.error(f'{e}')
                missing_packages.append(str(req))  # 保留原始输入字符串用于安装
        
        if missing_packages:
            self.install_packages(missing_packages, cmds)

    def install_packages(self, packages, cmds=''):
        try:
            LOGGER.warning(
                f'\nMissing packages: {", ".join(packages)}\nAttempting installation...'
            )
            # 构造 pip 命令参数（保持原有逻辑）
            pip_args = ['install', '--no-cache-dir'] + packages + cmds.split()
            # 调用 uv pip 安装（保持原有逻辑）
            subprocess.check_call(['uv', 'pip'] + pip_args)
            LOGGER.info('All the missing packages were installed successfully')
        except Exception as e:
            LOGGER.error(f'Failed to install packages: {e}')
            raise RuntimeError(f'Failed to install packages: {e}')