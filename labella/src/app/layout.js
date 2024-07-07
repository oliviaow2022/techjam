import { Inter } from "next/font/google";
import "./globals.css";
import { Toaster } from "react-hot-toast";

const inter = Inter({ subsets: ["latin"] });

export const metadata = {
  title: "Labella",
  description: "Data Labelling with Active Learning",
};

export default function RootLayout({ children }) {
  return (
    <html lang="en">
      <Toaster position="top-right" reverseOrder={false} />
      <body className={inter.className}>{children}</body>
    </html>
  );
}
