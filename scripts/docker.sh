user_name="wangtianshu"
project_name=${PWD##*/}
docker_registry="docker.cipsup.cn"

echo | docker login $docker_registry &>/dev/null || docker login $docker_registry -u $user_name
docker build --rm -t $user_name/$project_name .
docker tag $user_name/$project_name $docker_registry/$user_name/$project_name
docker push $docker_registry/$user_name/$project_name
