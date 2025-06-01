// frontend/src/components/Login.jsx

import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext'; 

const Login = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState(''); // Error state should always be a string or null
  const navigate = useNavigate();
  const { login } = useAuth();

  const handleLogin = async (e) => {
    e.preventDefault();
    setError(''); // Clear previous errors
    try {
      const userData = await login(email, password); 
      
      if (userData && userData.is_admin) {
        navigate('/admin');
      } else {
        navigate('/dashboard');
      }

    } catch (err) {
      console.error("Login API Error:", err.response || err); // Log the full error for debugging
      let errorMessage = 'An unexpected error occurred. Please try again.'; // Default generic error

      if (err.response && err.response.data) {
        if (typeof err.response.data.detail === 'string') {
          errorMessage = err.response.data.detail;
        } else if (Array.isArray(err.response.data.detail) && err.response.data.detail.length > 0) {
          // If detail is an array (e.g., Pydantic validation errors), take the first message
          // You might want to format this more nicely if there can be multiple messages
          const firstError = err.response.data.detail[0];
          if (firstError.msg) {
            errorMessage = firstError.msg;
          } else {
            errorMessage = 'Invalid input. Please check your details.';
          }
        } else if (typeof err.response.data.detail === 'object' && err.response.data.detail !== null) {
            // If detail is an object, try to stringify or pick a known field
            // This is a basic fallback; ideally, you'd know the structure
            errorMessage = JSON.stringify(err.response.data.detail); 
            // Or more specific: if (err.response.data.detail.message) errorMessage = err.response.data.detail.message;
        } else if (err.response.status === 401) {
            errorMessage = 'Invalid email or password.';
        } else if (err.response.status === 400) {
            errorMessage = 'Invalid request. Please check your input.';
        }
        // Add more specific status code checks if needed
      } else if (err.message) {
        // Network errors or other errors without a response object
        errorMessage = err.message;
      }
      
      setError(errorMessage);
    }
  };

  // ... rest of your JSX ...
  return (
    <div className="min-h-screen flex items-center justify-center relative">
      <video
        autoPlay
        loop
        muted
        className="absolute inset-0 w-full h-full object-cover"
      >
        <source src="/assets/jesa-promo-video.mp4" type="video/mp4" />
      </video>
      <div className="absolute inset-0 bg-blue-900 opacity-50"></div>
      <div className="relative bg-white p-8 rounded-lg shadow-lg max-w-md w-full">
        <img src="/assets/jesaImage.jpeg" alt="JESA Logo" className="mx-auto mb-4 h-16" />
        <h2 className="text-2xl font-bold text-center text-blue-600 mb-6">Sign In</h2>
        <form onSubmit={handleLogin}>
          <div className="mb-4">
            <label className="block text-gray-700 mb-2" htmlFor="email">Email</label>
            <input
              type="email"
              id="email"
              value={email}
              onChange={(e) => setEmail(e.target.value)}
              className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              required
              autoComplete="email"
            />
          </div>
          <div className="mb-6">
            <label className="block text-gray-700 mb-2" htmlFor="password">Password</label>
            <input
              type="password"
              id="password"
              value={password}
              onChange={(e) => setPassword(e.target.value)}
              className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
              required
              autoComplete="current-password"
            />
          </div>
          {error && <p className="text-red-500 text-center mb-4">{error}</p>}
          <button
            type="submit"
            className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 transition"
          >
            Sign In
          </button>
        </form>
        <div className="mt-4 text-center">
          <a href="/signup" className="text-blue-600 hover:underline">Sign Up</a>
          <span className="mx-2">|</span>
          <a href="/reset-password" className="text-blue-600 hover:underline">Forgot Password?</a>
        </div>
      </div>
    </div>
  );
};

export default Login;