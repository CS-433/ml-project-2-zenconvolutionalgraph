Start a new tmux session (-s session):
```
tmux new -s prova_session
```

Run Jupyter Notebook: Start the Jupyter Notebook server:
- `jupyter notebook` starts the Jupyter server, enabling you to interact with your notebooks via a web interface.
```
jupyter notebook --no-browser --port=8888
```

Access the notebook remotely (from your local machine)
`888:localhost:8888` maps your local machine’s port 8888 to the server’s port 8888, but on part 8888 in the server we put the jupyer noteook witht he previous command.
```
ssh -L 8888:localhost:8888 dalai@stiitsrv21.epfl.ch
```

Open a browser on your local machine and go to:
`http://localhost:8888` is mapped (like a pipe) to the poirt 8888 of server, that is where we have the notebook running.
```
http://localhost:8888
```

Detach the tmux session,The Jupyter Notebook will continue running in the background.
```
Press Ctrl+b, then press d.
```

Reattach to the tmux Session
```
tmux attach -t prova_session
```



