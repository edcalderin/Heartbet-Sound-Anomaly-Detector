```bash
docker build -t heartbet_detector_backend:01 -f backend_app.dockerfile .
kind load docker-image heartbet_detector_backend:01

kubectl apply -f kube_config/backend-deployment.yaml
kubectl get pod
kubectl port-forward backend-app-<replace-here> 8000:8000

kubectl apply -f kube-config/backend-service.yaml
kubectl get svc
kubectl port-forward service/backend-app 8000:8000

docker build -t heartbet_detector_frontend:01 -f frontend_app/Dockerfile frontend_app
kind load docker-image heartbet_detector_frontend:01

kubectl apply -f kube-config/frontend-deployment.yaml
kubectl get pod
kubectl port-forward frontend-app-<replace-here> 8501:8501

kubectl apply -f kube-config/frontend-service.yaml
kubectl get svc
kubectl port-forward service/frontend-app 9000:80
```

Go to `http://localhost:9000` to visualize the frontend application