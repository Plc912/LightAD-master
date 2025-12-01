#!/bin/bash
# LightAD MCP Server 启动脚本 (Linux/Mac)

echo "========================================"
echo "LightAD MCP Server 启动脚本"
echo "========================================"
echo ""

# 设置并发数
export LIGHTAD_MAX_CONCURRENT=2

echo "检查依赖..."
python3 -c "import fastmcp" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "[错误] 未安装 fastmcp，正在安装..."
    pip3 install fastmcp
fi

echo ""
echo "正在启动 MCP 服务器..."
echo "端口: 2224"
echo "并发数: $LIGHTAD_MAX_CONCURRENT"
echo ""
echo "按 Ctrl+C 停止服务器"
echo ""

python3 lightad_mcp_server.py

