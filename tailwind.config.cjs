/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {},
	},
	plugins: [require("@tailwindcss/typography"),require("daisyui")],
	daisyui: {
		themes: true, // true: all themes | false: only light + dark | array: specific themes like this ["light", "dark", "cupcake"]
		lightTheme: "retro", // name of the light theme
		darkTheme: "synthwave", // name of one of the included themes for dark mode
		logs: true, // Shows info about daisyUI version and used config in the console when building your CSS
	  }
}
