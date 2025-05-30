$virtualenv = "buildenv"

# check for python and pip in PATH

$pythonPath = Get-Command "python" -ErrorAction SilentlyContinue
if ($pythonPath -eq $null)
{
   throw "Unable to find python in PATH"
}

Write-Host "python found in " $pythonPath.Definition

$pipPath = Get-Command "pip" -ErrorAction SilentlyContinue
if ($pipPath -eq $null)
{
    # try to add the Python Scripts folder to the path
    $pythonPath = Get-Command "python" | Select-Object -ExpandProperty Definition | Split-Path
    $env:Path += ";$pythonPath\Scripts"

    # retry
    $pipPath = Get-Command "pip" -ErrorAction SilentlyContinue
    if ($pipPath -eq $null)
    {
        throw "Unable to find pip in PATH"
    }
}

Write-Host "pip found in " $pipPath.Definition

Write-Host "WIX env var set to " $env:WIX

Write-Host ""
Write-Host "==========================================="
Write-Host "Cleaning up"
Write-Host "==========================================="

Remove-Item $virtualenv -Recurse -ErrorAction Ignore
Remove-Item "build" -Recurse -ErrorAction Ignore
Remove-Item "dist" -Recurse -ErrorAction Ignore

Write-Host ""
Write-Host "==========================================="
Write-Host "Making sure pip is up-to-date"
Write-Host "==========================================="

& python -m pip install --upgrade pip

Write-Host ""
Write-Host "==========================================="
Write-Host "Making sure setuptools is up-to-date (for compiler compatibility)"
Write-Host "==========================================="

& pip install --upgrade setuptools

Write-Host ""
Write-Host "==========================================="
Write-Host "Installing virtualenv"
Write-Host "==========================================="

& pip install -U virtualenv

Write-Host ""
Write-Host "==========================================="
Write-Host "Creating a virtualenv"
Write-Host "==========================================="

& virtualenv $virtualenv

Write-Host ""
Write-Host "==========================================="
Write-Host "Activating the virtualenv"
Write-Host "==========================================="

& "$virtualenv\Scripts\activate"

Write-Host ""
Write-Host "==========================================="
Write-Host "Installing dependencies"
Write-Host "==========================================="

& pip install .

Write-Host ""
Write-Host "==========================================="
Write-Host "Building Cython extensions"
Write-Host "==========================================="

& python setup.py build_ext --inplace

Write-Host ""
Write-Host "==========================================="
Write-Host "Packaging with pyinstaller"
Write-Host "==========================================="

& pyinstaller liumotion.spec -y --log-level=DEBUG

Write-Host ""
Write-Host "==========================================="
Write-Host "Archiving the package as a zip file"
Write-Host "==========================================="

Compress-Archive -Path .\dist\liumotion\* -DestinationPath .\dist\liumotion.zip

Write-Host ""
Write-Host "==========================================="
Write-Host "Read version from file"
Write-Host "==========================================="
$initFileContent = Get-Content "liumotion/__init__.py"
$version = $initFileContent | Select-String '__version__ = \"([\d\.]+)\"' | Foreach-Object {$_.Matches.Groups[1].Value} 

Write-Host $version

Write-Host ""
Write-Host "==========================================="
Write-Host "Build MSI with WiX"
Write-Host "==========================================="

& "$env:WIX/bin/heat.exe" dir "dist/liumotion" -cg liumotionFiles -gg -scom -sreg -sfrag -srd -dr INSTALLFOLDER -out "dist/liumotionFilesFragment.wxs"
& "$env:WIX/bin/candle.exe" installer/liumotion.wxs -dVersion="$version" -o dist/wixobj/ -arch x64
& "$env:WIX/bin/candle.exe" dist/liumotionFilesFragment.wxs -o dist/wixobj/
& "$env:WIX/bin/light.exe" -ext WixUIExtension -cultures:en-us -b dist/liumotion dist/wixobj\*.wixobj -o "dist/liumotion-$version.msi"

# Installer can be tested with:
#    msiexec /i dist\liumotion-0.38.msi /l*v MyLogFile.txt
# for uninstall:
#    msiexec /x dist\liumotion-0.38.msi

Write-Host ""
Write-Host "==========================================="
Write-Host "Build appx package"
Write-Host "==========================================="

Copy-Item -Path .\dist\liumotion -Destination .\dist\liumotion-appx -Recurse
Copy-Item -Path resources\images\liumotion.iconset\icon_512x512.png -Destination .\dist\liumotion-appx\icon_512x512.png

# apply version to appxmanifest.xml and save it to the dist folder
$xml = [xml](Get-Content .\installer\appxmanifest.xml)
$ns = New-Object System.Xml.XmlNamespaceManager($xml.NameTable)
$ns.AddNamespace("ns", $xml.DocumentElement.NamespaceURI)
$package = $xml.SelectSingleNode("//ns:Package", $ns)
$package.Identity.Version = "$version.0.0"
$xml.Save(".\dist\liumotion-appx\appxmanifest.xml")

MakeAppx pack /v /d .\dist\liumotion-appx /p ".\dist\liumotion-$version.appx"
