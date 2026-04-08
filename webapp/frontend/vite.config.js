import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

export default defineConfig({
  plugins: [
    react({
      babel: {
        plugins: [["babel-plugin-react-compiler"]],
      },
    }),
  ],
  server: {
    proxy: {
      "/api": "http://127.0.0.1:9000",
      "/uploads": "http://127.0.0.1:9000",
      "/assets": "http://127.0.0.1:9000",
    },
  },
});
