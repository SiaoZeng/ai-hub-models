#!/bin/bash
# ---------------------------------------------------------------------
# Copyright (c) 2025 Qualcomm Technologies, Inc. and/or its subsidiaries.
# SPDX-License-Identifier: BSD-3-Clause
# ---------------------------------------------------------------------
set -e

if [ "$(id -u)" -eq 0 ]; then
  SUDO=""
else
  SUDO="sudo"
fi

echo "=== Installing system dependencies ==="
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq \
  ca-certificates \
  curl \
  git \
  libgl1-mesa-glx \
  libglib2.0-0 \
  libgomp1 \
  libsm6 \
  libxext6 \
  libxrender-dev \
  lsb-release \
  python3 \
  python3-pip \
  python3-venv \
  software-properties-common \
  unzip \
  zip \
  ffmpeg

echo "=== Configuring git ==="
git config --global --add safe.directory "$GITHUB_WORKSPACE"

echo "=== Installing Python 3.10 ==="
$SUDO add-apt-repository -y ppa:deadsnakes/ppa
$SUDO apt-get update -qq
$SUDO apt-get install -y -qq python3.10 python3.10-venv python3.10-dev
$SUDO update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.10 1
$SUDO ln -sf /usr/bin/python3.10 /usr/bin/python
python3 -m pip install --upgrade pip

echo "=== Installing AWS CLI ==="
curl -fsSL "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o /tmp/awscliv2.zip
unzip -q /tmp/awscliv2.zip -d /tmp
$SUDO /tmp/aws/install
rm -rf /tmp/awscliv2.zip /tmp/aws

echo "=== Installing uv 0.6.14 ==="
curl -fsSL https://github.com/astral-sh/uv/releases/download/0.6.14/uv-installer.sh | $SUDO env UV_UNMANAGED_INSTALL="/usr/local/bin" bash

echo "=== Pod setup complete ==="
