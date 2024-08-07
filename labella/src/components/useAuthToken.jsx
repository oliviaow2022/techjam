import { useSelector } from 'react-redux';

export default function useAuthToken() {
  const token = useSelector((state) => state.auth.jwtToken);
  return token;
};
