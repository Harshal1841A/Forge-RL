import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./src/pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/components/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/features/**/*.{js,ts,jsx,tsx,mdx}",
    "./src/app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        background: "#000000",
        foreground: "#FFFFFF",
      },
      backgroundImage: {
        'iridescent': 'linear-gradient(135deg, rgba(82,4,204,1) 0%, rgba(229,93,135,1) 50%, rgba(95,195,228,1) 100%)',
      },
    },
  },
  plugins: [],
};
export default config;
