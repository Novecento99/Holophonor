# -*- mode: python -*-

import os
import platform
import sys

# Add the path to the Holophonor module
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

import Holophonor  # for the version number
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

pathex = []
if platform.system() == "Windows":
  from PyInstaller.compat import getsitepackages
  pathex += [os.path.join(x, 'PyQt5', 'Qt', 'bin') for x in getsitepackages()]

a = Analysis(['main.py'],
             pathex=pathex,
             binaries=[],
             datas=collect_data_files('Holophonor'),
             hiddenimports=[],
             hookspath=[],
             runtime_hooks=[],
             excludes=[],
             win_no_prefer_redirects=False,
             win_private_assemblies=False,
             cipher=block_cipher)

pyz = PYZ(a.pure, a.zipped_data,
             cipher=block_cipher)

exe = EXE(pyz,
          a.scripts,
          exclude_binaries=True,
          name='Holophonor',
          debug=False,
          strip=False,
          upx=False,
          console=False,
          icon="resources/images/Holophonor.ico")

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='Holophonor')

app = BUNDLE(coll,
         name='Holophonor.app',
         icon='resources/images/Holophonor.icns',
         bundle_identifier="org.example.Holophonor",
         version=Holophonor.__version__,
         info_plist={
            'NSMicrophoneUsageDescription': 'Holophonor reads from the audio inputs to show visualizations',
            'CFBundleVersion': Holophonor.__version__
         })
