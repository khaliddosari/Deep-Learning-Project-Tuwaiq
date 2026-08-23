import type { Metadata } from "next";
import "./globals.css";

const title = "CIFAR-10 with Dense Networks Only";
const description =
  "Deep Learning capstone: 20 controlled experiments classifying CIFAR-10 with fully connected networks and no convolution — before optimization, after optimization, insights, and why the ceiling is architectural.";

export const metadata: Metadata = {
  title,
  description,
  icons: { icon: "/favicon.svg", shortcut: "/favicon.svg" },
  openGraph: { title, description, type: "website" },
};

export default function RootLayout({ children }: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en">
      <body>{children}</body>
    </html>
  );
}
