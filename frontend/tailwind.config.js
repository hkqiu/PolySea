/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,tsx}"],
  theme: {
    extend: {
      fontFamily: {
        sans: [
          "-apple-system",
          "BlinkMacSystemFont",
          '"SF Pro SC"',
          '"SF Pro Text"',
          '"PingFang SC"',
          '"Helvetica Neue"',
          "Helvetica",
          "Arial",
          "sans-serif",
        ],
      },
      colors: {
        apple: {
          blue: "#0071e3",
          gray: "#86868b",
          label: "#1d1d1f",
          subtle: "#6e6e73",
          fill: "#f5f5f7",
          border: "#d2d2d7",
        },
      },
    },
  },
  plugins: [],
};
