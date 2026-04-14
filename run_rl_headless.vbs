Dim fso, shell, root, cmd
Set fso = CreateObject("Scripting.FileSystemObject")
Set shell = CreateObject("WScript.Shell")

root = fso.GetParentFolderName(WScript.ScriptFullName)
cmd = "powershell.exe -NoProfile -ExecutionPolicy Bypass -File """ & root & "\headless_loop.ps1"" --disable-collector"

' 0 = hidden window, False = do not wait
shell.Run cmd, 0, False
