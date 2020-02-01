# UGB

Universal Game Bot with deep learning

## 设计思路

- 把画面截图裁剪成N个固定大小`image_size`的方块
- 应用TRIPLELOSS：把方块和模板图（缩放至image_size）分别输入CNN，
