Start-Service -Name "SakuraFrpService"
Start-Process -FilePath "python" -ArgumentList "server.py" -WorkingDirectory E:\Workspace\SoundHorizon\HandGesture 
Start-Process -FilePath "python" -ArgumentList "gpu-server.py" -WorkingDirectory E:\Workspace\SoundHorizon\ODAS 
Start-Process -FilePath "python" -ArgumentList "server.py"