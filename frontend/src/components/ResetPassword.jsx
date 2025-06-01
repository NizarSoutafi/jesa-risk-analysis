import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';

const ResetPassword = () => {
  const [email, setEmail] = useState('');
  const [code, setCode] = useState('');
  const [newPassword, setNewPassword] = useState('');
  const [step, setStep] = useState('email');
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const navigate = useNavigate();

  const handleSendCode = async (e) => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:8000/api/auth/reset-password/request', { email });
      setStep('code');
    } catch (err) {
      setError('Invalid email or error sending code');
    }
  };

  const handleResetPassword = async (e) => {
    e.preventDefault();
    try {
      await axios.post('http://localhost:8000/api/auth/reset-password/verify', {
        email,
        code,
        new_password: newPassword,
      });
      setSuccess('Password reset! Please sign in.');
      setTimeout(() => navigate('/login'), 2000);
    } catch (err) {
      setError('Invalid code or error resetting password');
    }
  };

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
        <h2 className="text-2xl font-bold text-center text-blue-600 mb-6">Reset Password</h2>
        {step === 'email' ? (
          <form onSubmit={handleSendCode}>
            <div className="mb-6">
              <label className="block text-gray-700 mb-2" htmlFor="email">Email</label>
              <input
                type="email"
                id="email"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                required
              />
            </div>
            {error && <p className="text-red-500 text-center mb-4">{error}</p>}
            <button
              type="submit"
              className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 transition"
            >
              Send Code
            </button>
          </form>
        ) : (
          <form onSubmit={handleResetPassword}>
            <div className="mb-4">
              <label className="block text-gray-700 mb-2" htmlFor="code">Verification Code</label>
              <input
                type="text"
                id="code"
                value={code}
                onChange={(e) => setCode(e.target.value)}
                className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                required
              />
            </div>
            <div className="mb-6">
              <label className="block text-gray-700 mb-2" htmlFor="newPassword">New Password</label>
              <input
                type="password"
                id="newPassword"
                value={newPassword}
                onChange={(e) => setEmail(e.target.value)}
                className="w-full p-3 border rounded-lg focus:outline-none focus:ring-2 focus:ring-blue-600"
                required
              />
            </div>
            {error && <p className="text-red-500 text-center mb-4">{error}</p>}
            {success && <p className="text-green-500 text-center mb-4">{success}</p>}
            <button
              type="submit"
              className="w-full bg-blue-600 text-white p-3 rounded-lg hover:bg-blue-700 transition"
            >
              Reset Password
            </button>
          </form>
        )}
        <div className="mt-4 text-center">
          <a href="/login" className="text-blue-600 hover:underline">Back to Sign In</a>
        </div>
      </div>
    </div>
  );
};

export default ResetPassword;