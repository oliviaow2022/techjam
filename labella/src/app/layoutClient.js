"use client"; // This makes this component a Client Component

import { Toaster } from "react-hot-toast";
import dynamic from "next/dynamic";

const ReduxProvider = dynamic(() => import("@/store/redux-provider"), {
  ssr: false,
});

export default function RootLayoutClient({ children }) {
  return (
    <div>
      <Toaster position="top-right" reverseOrder={false} />
      <ReduxProvider>{children}</ReduxProvider>
    </div>
  );
}
