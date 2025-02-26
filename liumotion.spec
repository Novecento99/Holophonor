# -*- mode: python -*-

import os
import platform

import liumotion  # for the version number
from PyInstaller.utils.hooks import collect_data_files

block_cipher = None

pathex = []
if platform.system() == "Windows":
  from PyInstaller.compat import getsitepackages
  pathex += [os.path.join(x, 'PyQt5', 'Qt', 'bin') for x in getsitepackages()]

a = Analysis(['main.py'],
             pathex=pathex,
             binaries=[],
             datas=collect_data_files('liumotion'),
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
          name='liumotion',
          debug=False,
          strip=False,
          upx=False,
          console=False,
          icon="resources/images/liumotion.ico")

coll = COLLECT(exe,
               a.binaries,
               a.zipfiles,
               a.datas,
               strip=False,
               upx=False,
               name='liumotion')

app = BUNDLE(coll,
         name='liumotion.app',
         icon='resources/images/liumotion.icns',
         bundle_identifier="org.example.liumotion",
         version=liumotion.__version__,
         info_plist={
            'NSMicrophoneUsageDescription': 'LiuMotion reads from the audio inputs to show visualizations',
            'CFBundleVersion': liumotion.__version__
         })
