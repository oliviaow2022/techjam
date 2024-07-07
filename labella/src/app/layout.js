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
      <body className={inter.className}>
        <div>
          <Toaster position="top-right" reverseOrder={false} />
        </div>
        {children}
      </body>
    </html>
  );
}
