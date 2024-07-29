const CopyWebpackPlugin = require('copy-webpack-plugin');
const path = require('path');

module.exports = function override(config, env) {
  config.plugins = [
    ...config.plugins,
    new CopyWebpackPlugin({
      patterns: [
        {
          from: path.resolve(__dirname, 'node_modules/onnxruntime-web/dist/*.wasm'),
          to: 'static/js/[name][ext]'
        },
      ],
    }),
  ];

  return config;
};
