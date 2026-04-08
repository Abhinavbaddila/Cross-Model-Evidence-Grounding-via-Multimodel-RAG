import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const repoBase = "/Cross-Model-Evidence-Grounding-via-Multimodel-RAG/";
const isGithubPagesBuild = process.env.GITHUB_ACTIONS === "true";

export default defineConfig({
  base: isGithubPagesBuild ? repoBase : "/",
  build: {
    assetsDir: "app-assets",
  },
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
