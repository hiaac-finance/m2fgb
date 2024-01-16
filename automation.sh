docker build -t xgb:$USER -f Dockerfile \
    --build-arg OUTSIDE_GROUP=`/usr/bin/id -ng $USER` \
    --build-arg OUTSIDE_GID=`/usr/bin/id -g $USER` \
    --build-arg OUTSIDE_USER=$USER \
    --build-arg OUTSIDE_UID=$UID .

docker run -it \
    --userns=host \
    --name xgb \
    -v /work/$USER:/work/$USER \
    xgb:$USER \
    /bin/bash

# docker exec -ti -u $USER xgb bash