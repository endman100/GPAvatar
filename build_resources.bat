@REM @echo off

REM 下載文件
@REM powershell -Command "Invoke-WebRequest -Uri 'https://github.com/xg-chu/GPAvatar/releases/download/Resources/resources.tar' -OutFile './resources.tar'"

REM 檢查文件完整性
powershell -Command "Get-FileHash -Algorithm MD5 './resources.tar' | Format-List"
echo "1d386517baa670307243c3fd45494a53 Please check if the md5sum is correct"

REM 解壓縮文件
powershell -Command "tar -xvf ./resources.tar"

REM 移動文件
move resources\examples demos\
move resources\drivers demos\

move resources\main_params\* core\libs\FLAME\assets\
move resources\track_params\* core\libs\lightning_track\engines\FLAME\assets\
move resources\matting\* core\libs\lightning_track\engines\human_matting\assets\
move resources\emoca\* core\libs\lightning_track\engines\emoca\assets\
move resources\mica_base\* core\libs\lightning_track\engines\mica\assets\

@REM REM 刪除不需要的文件和文件夾
@REM rmdir /s /q resources
@REM del resources.tar

@REM @echo on
