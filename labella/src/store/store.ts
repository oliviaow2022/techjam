import { combineReducers, configureStore } from "@reduxjs/toolkit";
import { authReducer } from "@/store/authSlice";
import { persistReducer, persistStore } from "redux-persist";
import storageSession from 'redux-persist/lib/storage/session'

// configure which key we want to persist
const authPersistConfig = {
  key: "auth",
  storage: storageSession,
  whitelist: ["authState"],
};

const rootReducer = combineReducers({
  auth: persistReducer(authPersistConfig, authReducer),
});

export const store = configureStore({
  reducer: rootReducer,
  middleware: (getDefaultMiddleware) =>
    getDefaultMiddleware({ serializableCheck: false }),
});

export const persistor = persistStore(store)