from flask import Flask, request, jsonify, send_from_directory, session
from flask_cors import CORS
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
import json
import os
import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time
import re
import logging
from zhipuai import ZhipuAI
from dotenv import load_dotenv
import uuid
from datetime import datetime, timedelta
import shutil

# ==================== 配置加载与初始化 ====================
load_dotenv()

# 日志配置（增加请求ID和详细上下文）
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(request_id)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# 应用配置（从环境变量读取，默认值兜底）
APP_HOST = os.getenv("APP_HOST", "0.0.0.0")
APP_PORT = int(os.getenv("APP_PORT", 5186))
APP_DEBUG = os.getenv("APP_DEBUG", "False").lower() == "true"
ZHIPU_API_KEY = os.getenv("ZHIPU_API_KEY")
ZHIPU_MODEL = os.getenv("ZHIPU_MODEL", "glm-4-flash")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")
FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "xrd-frontend")
# 图片保留时间（小时）
IMAGE_EXPIRE_HOURS = int(os.getenv("IMAGE_EXPIRE_HOURS", 24))

# 校验关键配置
if not ZHIPU_API_KEY:
    raise ValueError("未找到ZHIPU_API_KEY，请在.env文件中配置")

# ==================== 应用初始化 ====================
app = Flask(__name__)
# 启用Session（用于存储用户唯一结果ID）
app.secret_key = os.getenv("SECRET_KEY", str(uuid.uuid4()))  # 生产环境务必配置SECRET_KEY
CORS(app, supports_credentials=True)  # 允许跨域携带Cookie

# 接口限流（防止恶意请求）
limiter = Limiter(
    get_remote_address,
    app=app,
    default_limits=["20 per minute", "1 per second"],
    storage_uri="memory://"
)

# ==================== 数据定义 ====================
# 标准XRD库
STANDARD_XRD = {
    "ZnO": {
        "2theta": [31.77, 34.42, 36.25, 47.54, 56.60, 62.86, 67.95],
        "intensity": [100, 60, 80, 40, 30, 25, 20],
        "name": "ZnO 标准图谱"
    },
    "TiO2": {
        "2theta": [25.28, 37.80, 48.05, 54.72, 62.70, 68.76],
        "intensity": [100, 30, 45, 15, 20, 18],
        "name": "TiO2 标准图谱"
    },
    "CeO2": {
        "2theta": [28.55, 33.08, 47.48, 56.33, 59.09, 69.41],
        "intensity": [100, 45, 50, 25, 20, 15],
        "name": "CeO2 标准图谱"
    }
}

# 结果存储（key: 唯一ID, value: 结果字典）
result_store = {}


# ==================== 工具函数 ====================
def get_request_id():
    """生成/获取请求ID（用于日志追踪）"""
    if not session.get("request_id"):
        session["request_id"] = str(uuid.uuid4())
    return session["request_id"]


def clean_expired_images():
    """清理过期图片文件（后台定时/按需执行）"""
    try:
        now = datetime.now()
        for filename in os.listdir(STATIC_DIR):
            if filename.startswith(("xrd_standard_", "xrd_ai_")):
                file_path = os.path.join(STATIC_DIR, filename)
                file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path))
                if now - file_mtime > timedelta(hours=IMAGE_EXPIRE_HOURS):
                    os.remove(file_path)
                    logger.info(f"清理过期图片: {filename}", extra={"request_id": get_request_id()})

        # 清理过期结果
        expired_ids = []
        for res_id, res_data in result_store.items():
            if now - res_data["create_time"] > timedelta(hours=IMAGE_EXPIRE_HOURS):
                expired_ids.append(res_id)
        for res_id in expired_ids:
            del result_store[res_id]
            logger.info(f"清理过期结果: {res_id}", extra={"request_id": get_request_id()})
    except Exception as e:
        logger.error(f"清理过期文件失败: {str(e)}", extra={"request_id": get_request_id()})


def get_standard_xrd(sample_name):
    """根据样品名称匹配标准XRD数据"""
    for key in STANDARD_XRD:
        if key in sample_name:
            return STANDARD_XRD[key]
    return STANDARD_XRD["ZnO"]  # 默认返回ZnO


def call_zhipu_ai(sample, method):
    """调用智谱AI生成XRD数据（封装+降级策略）"""
    prompt = f"""你是材料XRD专家，请根据以下信息生成XRD图谱数据。
材料：{sample}
制备方法：{method}
要求：
1. 只返回JSON格式，不要任何解释文字
2. 必须包含2theta和intensity两个字段
3. 2theta范围在20-80度之间
4. intensity范围在0-100之间
5. 至少包含5个衍射峰
返回格式示例：
{{"2theta": [31.77, 34.42, 36.25, 47.54, 56.60], "intensity": [100, 60, 80, 40, 30]}}"""

    try:
        client = ZhipuAI(api_key=ZHIPU_API_KEY)
        response = client.chat.completions.create(
            model=ZHIPU_MODEL,
            messages=[{"role": "user", "content": prompt}],
            timeout=30
        )
        content = response.choices[0].message.content.strip()

        # 清理JSON格式
        if "```" in content:
            parts = content.split("```")
            for p in parts:
                p = p.strip()
                if p.startswith("{"):
                    content = p
                    break
                if p.lower().startswith("json"):
                    content = p[4:].strip()
                    break

        # 提取JSON核心部分
        start_idx = content.find("{")
        end_idx = content.rfind("}")
        if start_idx == -1 or end_idx == -1:
            raise ValueError("AI返回内容无有效JSON")
        content = content[start_idx:end_idx + 1]

        # 修复JSON格式问题
        content = content.replace("'", '"')
        content = re.sub(r'(\w+):', r'"\1":', content)
        ai_data = json.loads(content)

        # 数据校验
        if not isinstance(ai_data.get("2theta"), list) or not isinstance(ai_data.get("intensity"), list):
            raise ValueError("2theta/intensity不是数组")
        if len(ai_data["2theta"]) != len(ai_data["intensity"]) or len(ai_data["2theta"]) < 5:
            raise ValueError("数据长度不符合要求")
        if not all(20 <= t <= 80 for t in ai_data["2theta"]):
            raise ValueError("2theta超出20-80度范围")
        if not all(0 <= i <= 100 for i in ai_data["intensity"]):
            raise ValueError("intensity超出0-100范围")

        return ai_data
    except Exception as e:
        logger.error(f"AI调用失败: {str(e)}", extra={"request_id": get_request_id()})
        # 降级策略：返回标准数据的轻微扰动版本
        standard = get_standard_xrd(sample)
        np.random.seed(int(time.time()) % 1000)
        theta = [t + np.random.uniform(-0.5, 0.5) for t in standard["2theta"]]
        intensity = [i + np.random.uniform(-10, 10) for i in standard["intensity"]]
        intensity = [max(0, min(100, i)) for i in intensity]  # 限制范围
        return {"2theta": theta, "intensity": intensity}


def plot_xrd_image(x, y, title, sample, method, filename):
    """封装XRD绘图逻辑"""
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False

    fig, ax = plt.subplots(1, 1, figsize=(12, 5), dpi=100)
    ax.plot(x, y, label=title, color="#007bff" if "标准" in title else "#dc3545", linewidth=2)
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel("2θ (°)", fontsize=12)
    ax.set_ylabel("Intensity (a.u.)", fontsize=12)
    ax.legend(loc='best')
    ax.grid(alpha=0.3)
    plt.suptitle(f"样品: {sample} | 方法: {method}", fontsize=12, y=0.98)
    plt.tight_layout()

    # 保存图片
    os.makedirs(STATIC_DIR, exist_ok=True)
    img_path = os.path.join(STATIC_DIR, filename)
    plt.savefig(img_path, bbox_inches="tight", dpi=150)
    plt.close()
    return f"/static/{filename}"


# ==================== 路由函数 ====================
@app.route('/')
def index():
    return send_from_directory(FRONTEND_DIR, 'index.html')


@app.route('/result')
def result():
    return send_from_directory(FRONTEND_DIR, 'result.html')


@app.route('/<filename>')
def serve_frontend_static(filename):
    """安全的静态文件路由（限制可访问的文件类型）"""
    safe_files = ['style.css', 'script.js']
    if filename in safe_files:
        return send_from_directory(FRONTEND_DIR, filename)
    # 限制静态文件类型为图片
    if filename.endswith(('.png', '.jpg', '.jpeg', '.gif')):
        return send_from_directory(STATIC_DIR, filename)
    return jsonify({"success": False, "error": "禁止访问"}), 403


@app.route("/api/generate", methods=["POST"])
@limiter.limit("5 per minute")  # 单独限制生成接口
def api_generate():
    """生成XRD图谱接口（优化版）"""
    request_id = get_request_id()
    # 定期清理过期文件（每10次请求执行一次，也可改用定时任务）
    if len(result_store) % 10 == 0:
        clean_expired_images()

    try:
        params = request.json
        if not params:
            return jsonify({"success": False, "error": "请求参数不能为空"})

        # 严格参数校验
        sample = params.get("样品名称", "").strip()
        method = params.get("制备方法", "").strip()
        if not sample:
            return jsonify({"success": False, "error": "样品名称不能为空"})

        # 1. 获取标准数据
        standard_data = get_standard_xrd(sample)
        # 2. 调用AI生成数据
        ai_data = call_zhipu_ai(sample, method)
        # 3. 生成平滑曲线
        x = np.linspace(10, 80, 1000)
        y_std = np.zeros_like(x)
        for t, i in zip(standard_data["2theta"], standard_data["intensity"]):
            y_std += np.exp(-((x - t) / 0.25) ** 2) * i
        y_ai = np.zeros_like(x)
        for t, i in zip(ai_data["2theta"], ai_data["intensity"]):
            y_ai += np.exp(-((x - t) / 0.25) ** 2) * i

        # 4. 生成图片
        timestamp = int(time.time())
        img_std = plot_xrd_image(x, y_std, standard_data["name"], sample, method, f"xrd_standard_{timestamp}.png")
        img_ai = plot_xrd_image(x, y_ai, "AI生成", sample, method, f"xrd_ai_{timestamp}.png")

        # 5. 存储结果（唯一ID关联）
        result_id = str(uuid.uuid4())
        result_store[result_id] = {
            "image_standard": img_std,
            "image_ai": img_ai,
            "params": params,
            "create_time": datetime.now()
        }
        # 将结果ID返回给前端
        return jsonify({
            "success": True,
            "message": "生成成功",
            "result_id": result_id
        })
    except ValueError as e:
        logger.error(f"参数/数据错误: {str(e)}", extra={"request_id": request_id})
        return jsonify({"success": False, "error": f"数据错误: {str(e)}"})
    except Exception as e:
        logger.error(f"生成失败: {str(e)}", extra={"request_id": request_id})
        return jsonify({"success": False, "error": f"生成失败: {str(e)}"})


@app.route("/api/get-result")
def api_get_result():
    """获取结果接口（通过result_id）"""
    result_id = request.args.get("result_id")
    if not result_id or result_id not in result_store:
        return jsonify({"success": False, "error": "结果不存在或已过期"})
    return jsonify({
        "success": True,
        "data": result_store[result_id]
    })


# ==================== 启动应用 ====================
if __name__ == "__main__":
    # 启动前清理一次过期文件
    clean_expired_images()
    app.run(host=APP_HOST, port=APP_PORT, debug=APP_DEBUG)
