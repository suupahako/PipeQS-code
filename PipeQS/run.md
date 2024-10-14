# 2docker通信
  #### 修改代码后挂载
  docker rm -f container1
  docker rm -f container2



  docker run --gpus all -it --name container1 -v ~/code/PipeGCN-original:/path/in/container --shm-size=8g cheng1016/bns-gcn
  cd /../path/in/container/
  python helper/quantization/setup.py install

  docker run --gpus all -it --name container2 -v ~/code/PipeGCN-original:/path/in/container --shm-size=8g cheng1016/bns-gcn
  cd /../path/in/container/
  python helper/quantization/setup.py install 

  #### 查看docker ip地址
  docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container1
  #### 测试通信
  docker exec -it container1 bash -c "apt-get update && apt-get install -y iputils-ping"
  docker exec container1 ping -c 4 172.17.0.5
  #### setup




# 4docker通信
    #### 修改代码后挂载
  docker rm -f container1
  docker rm -f container2
  docker rm -f container3
  docker rm -f container4

  docker run --gpus all -it --name container1 -v ~/code/BNS-GCN-QT:/path/in/container --shm-size=8g cheng1016/bns-gcn

  docker run --gpus all -it --name container2 -v ~/code/BNS-GCN-QT:/path/in/container --shm-size=8g cheng1016/bns-gcn

  docker run --gpus all -it --name container3 -v ~/code/BNS-GCN-QT:/path/in/container --shm-size=8g cheng1016/bns-gcn
  
  docker run --gpus all -it --name container4 -v ~/code/BNS-GCN-QT:/path/in/container --shm-size=8g cheng1016/bns-gcn
  
  #### 查看docker ip地址
  docker inspect -f '{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' container1
  #### 测试通信
  docker exec -it container1 bash -c "apt-get update && apt-get install -y iputils-ping"
  docker exec container1 ping -c 4 172.17.0.5
  #### setup
  cd /../path/in/container/
  python helper/quantization/setup.py install