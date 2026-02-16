"""
同步标注组件：操作区与预览区使用 RAF 同帧渲染，消除偏移。
"""
import os
import streamlit.components.v1 as components

_COMPONENT_PATH = os.path.join(os.path.dirname(__file__), "frontend")


def synced_annotation(
    background_image,
    initial_drawing=None,
    drawing_mode="circle",
    width=600,
    height=450,
    image_width=None,
    image_height=None,
    key=None,
):
    """
    渲染同步标注组件：左侧操作区、右侧预览区，RAF 同帧更新。
    Args:
        background_image: PIL Image 或 numpy array
        initial_drawing: {"objects": [...]}
        drawing_mode: "circle" | "transform"
        width, height: 单个画布尺寸（两侧相同）
        key: Streamlit key
    Returns:
        {"objects": [...], "canvas_width": w, "canvas_height": h} 或 None
    """
    from io import BytesIO
    import base64

    # 将背景图转为 base64（JPEG 降低体积）
    if hasattr(background_image, "save"):
        bg = background_image.convert("RGB") if background_image.mode in ("RGBA", "P") else background_image
        buf = BytesIO()
        bg.save(buf, format="JPEG", quality=90, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()
    else:
        import numpy as np
        from PIL import Image
        arr = np.asarray(background_image)
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        img = Image.fromarray(arr.astype("uint8")).convert("RGB")
        buf = BytesIO()
        img.save(buf, format="JPEG", quality=90, optimize=True)
        b64 = base64.b64encode(buf.getvalue()).decode()

    iw = image_width if image_width is not None else width
    ih = image_height if image_height is not None else height
    args = {
        "backgroundImageBase64": b64,
        "width": int(width),
        "height": int(height),
        "imageWidth": int(iw),
        "imageHeight": int(ih),
        "drawingMode": drawing_mode,
        "initialDrawing": initial_drawing or {"objects": [], "version": "4.0.0"},
    }
    result = components.declare_component("synced_annotation", path=_COMPONENT_PATH)(
        **args,
        key=key,
    )
    return result
