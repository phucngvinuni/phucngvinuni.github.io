/** @type {import('tailwindcss').Config} */
module.exports = {
	content: ['./src/**/*.{astro,html,js,jsx,md,mdx,svelte,ts,tsx,vue}'],
	theme: {
		extend: {},
	},
	plugins: [require("@tailwindcss/typography"),require("daisyui")],
	daisyui: {
		themes: [
			{
				linear: {
					"primary": "#5E6AD2",
					"secondary": "#8A8F98",
					"accent": "#C292EB",
					"neutral": "#1C1D21",
					"base-100": "#08090A",
					"base-200": "#141517",
					"base-300": "#1C1D21",
					"base-content": "#F7F8F8",
					"info": "#3ABFF8",
					"success": "#36D399",
					"warning": "#FBBD23",
					"error": "#F87272",
				},
			},
			"light",
			"dark",
		],
		darkTheme: "linear", // name of one of the included themes for dark mode
		logs: true, // Shows info about daisyUI version and used config in the console when building your CSS
	  }
}
