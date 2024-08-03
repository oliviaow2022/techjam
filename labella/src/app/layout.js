import { Inter } from "next/font/google";
import "./globals.css";
import RootLayoutClient from "./layoutClient";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Labella",
  description: "Data Labelling with Active Learning",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <body className={inter.className}>
        <RootLayoutClient>{children}</RootLayoutClient>
      </body>
    </html>
  );
}
