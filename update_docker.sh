#!/bin/bash

set -e

echo "🔧 Удаляем старую версию Docker..."
sudo apt remove -y docker docker-engine docker.io containerd runc || true

echo "📦 Обновляем APT и устанавливаем зависимости..."
sudo apt update
sudo apt install -y ca-certificates curl gnupg lsb-release

echo "🔑 Добавляем GPG ключ Docker..."
sudo mkdir -p /etc/apt/keyrings
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | \
  sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg

echo "📄 Добавляем Docker репозиторий..."
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] \
  https://download.docker.com/linux/ubuntu \
  $(lsb_release -cs) stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null

echo "🔄 Обновляем APT и устанавливаем Docker..."
sudo apt update
sudo apt install -y docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin

echo "✅ Проверка версии Docker:"
docker --version
docker compose version

echo "🚀 Готово! Docker обновлён и готов к работе."
