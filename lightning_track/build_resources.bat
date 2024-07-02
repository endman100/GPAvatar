
@REM REM 下載文件
@REM powershell -Command "Invoke-WebRequest -Uri 'https://github.com/xg-chu/lightning_track/releases/download/resources/resources.tar' -OutFile './resources.tar'"

REM 解壓縮文件
powershell -Command "tar -xvf './resources.tar'"

REM 移動文件
move resources\emoca\* engines\emoca\assets\
move resources\FLAME\* engines\FLAME\assets\
move resources\human_matting\* engines\human_matting\assets\
move resources\mica\* engines\mica\assets\

@REM REM 刪除不需要的文件和文件夾
@REM rmdir /s /q resources
@REM del resources.tar
