const express = require('express');
const path = require('path');
const app = express();

// 设置跨源策略的中间件
app.use((req, res, next) => {
  res.setHeader('Cross-Origin-Embedder-Policy', 'require-corp');
  res.setHeader('Cross-Origin-Opener-Policy', 'same-origin');
  next();
});

// 提供静态文件
app.use(express.static(path.join(__dirname, 'build')));

// 捕获所有路由，返回 React 应用的入口点
app.get('*', (req, res) => {
  res.sendFile(path.join(__dirname, 'build', 'index.html'));
});

// 启动服务器
const PORT = process.env.PORT || 3000;
app.listen(PORT, () => {
  console.log(`Server is running on port ${PORT}`);
});
