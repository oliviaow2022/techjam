import axios from 'axios';
import { useAuthToken } from './useAuthToken'; // Import the custom hook

const createApiClient = () => {
  const apiClient = axios.create({
    baseURL: process.env.NEXT_PUBLIC_API_ENDPOINT, // Your API base URL
    timeout: 10000, // Optional: Set a timeout for requests
    headers: {
      'Content-Type': 'application/json',
    }
  });

  apiClient.interceptors.request.use(
    (config) => {
      const token = useAuthToken; // Get the token using the custom hook
      if (token) {
        config.headers['Authorization'] = `Bearer ${token}`;
      }
      return config;
    },
    (error) => {
      return Promise.reject(error);
    }
  );

  return apiClient;
};

export default createApiClient;
