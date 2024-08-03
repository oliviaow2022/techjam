import { createSlice } from "@reduxjs/toolkit";
import type { PayloadAction } from "@reduxjs/toolkit";
export interface IAuthState {
  jwtToken: string;
  userId: string,
}

const initialState: IAuthState = {
  jwtToken: '',
  userId: '',
};

export const authSlice = createSlice({
  name: "auth",
  initialState,
  reducers: {
    setJwtToken: (state, action: PayloadAction<string>) => {
      state.jwtToken = action.payload;
    },
  },
});

export const { setJwtToken } = authSlice.actions;
export const authReducer = authSlice.reducer;