import React, { useState, useEffect } from 'react';
  import axios from 'axios';
  import { useAuth } from '../context/AuthContext';
  import { useNavigate } from 'react-router-dom';

  const AdminPanel = () => {
    const { token, logout } = useAuth();
    const [users, setUsers] = useState([]);
    const [error, setError] = useState('');
    const [success, setSuccess] = useState('');
    const [newUser, setNewUser] = useState({ name: '', email: '', password: '', isAdmin: false });
    const navigate = useNavigate();

    useEffect(() => {
      fetchUsers();
    }, []);

    const fetchUsers = async () => {
      try {
        const response = await axios.get('http://localhost:8000/api/admin/users', {
          headers: { Authorization: `Bearer ${token}` }
        });
        setUsers(response.data);
      } catch (err) {
        setError('Failed to fetch users');
        if (err.response?.status === 403) {
          logout();
          navigate('/login');
        }
      }
    };

    const addUser = async () => {
      if (!newUser.name || !newUser.email || !newUser.password) {
        setError('All fields are required');
        return;
      }
      try {
        await axios.post('http://localhost:8000/api/auth/signup', {
          name: newUser.name,
          email: newUser.email,
          password: newUser.password,
          is_admin: newUser.isAdmin
        }, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setSuccess('User added successfully');
        setError('');
        setNewUser({ name: '', email: '', password: '', isAdmin: false });
        fetchUsers(); // Refresh the user list
      } catch (err) {
        setError(err.response?.data?.detail || 'Failed to add user');
        setSuccess('');
      }
    };

    const deleteUser = async (userId) => {
      if (window.confirm('Are you sure you want to delete this user?')) {
        try {
          await axios.delete(`http://localhost:8000/api/admin/users/${userId}`, {
            headers: { Authorization: `Bearer ${token}` }
          });
          setSuccess('User deleted successfully');
          setError('');
          fetchUsers();
        } catch (err) {
          setError('Failed to delete user');
          setSuccess('');
        }
      }
    };

    const toggleAdminStatus = async (userId) => {
      try {
        await axios.put(`http://localhost:8000/api/admin/users/${userId}/toggle-admin`, {}, {
          headers: { Authorization: `Bearer ${token}` }
        });
        setSuccess('Admin status updated');
        setError('');
        fetchUsers();
      } catch (err) {
        setError('Failed to update admin status');
        setSuccess('');
      }
    };

    return (
      <div className="min-h-screen bg-gray-100 p-6">
        <div className="max-w-5xl mx-auto bg-white shadow-lg rounded-lg p-8">
          <div className="flex justify-between items-center mb-6">
            <h1 className="text-3xl font-bold text-gray-800">Admin Panel</h1>
            <div>
              <button
                onClick={() => navigate('/dashboard')}
                className="bg-blue-500 hover:bg-blue-600 text-white font-semibold py-2 px-4 rounded mr-2"
              >
                Back to Dashboard
              </button>
              <button
                onClick={logout}
                className="bg-red-500 hover:bg-red-600 text-white font-semibold py-2 px-4 rounded"
              >
                Logout
              </button>
            </div>
          </div>

          {/* Success/Error Messages */}
          {error && <p className="text-red-500 mb-4">{error}</p>}
          {success && <p className="text-green-500 mb-4">{success}</p>}

          {/* Add User Form */}
          <div className="mb-8">
            <h2 className="text-2xl font-semibold text-gray-700 mb-4">Add New User</h2>
            <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
              <input
                type="text"
                placeholder="Name"
                value={newUser.name}
                onChange={(e) => setNewUser({ ...newUser, name: e.target.value })}
                className="border rounded p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <input
                type="email"
                placeholder="Email"
                value={newUser.email}
                onChange={(e) => setNewUser({ ...newUser, email: e.target.value })}
                className="border rounded p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <input
                type="password"
                placeholder="Password"
                value={newUser.password}
                onChange={(e) => setNewUser({ ...newUser, password: e.target.value })}
                className="border rounded p-2 focus:outline-none focus:ring-2 focus:ring-blue-500"
              />
              <div className="flex items-center">
                <input
                  type="checkbox"
                  checked={newUser.isAdmin}
                  onChange={(e) => setNewUser({ ...newUser, isAdmin: e.target.checked })}
                  className="mr-2"
                />
                <label className="text-gray-700">Make Admin</label>
              </div>
            </div>
            <button
              onClick={addUser}
              className="mt-4 bg-green-500 hover:bg-green-600 text-white font-semibold py-2 px-4 rounded"
            >
              Add User
            </button>
          </div>

          {/* User List */}
          <h2 className="text-2xl font-semibold text-gray-700 mb-4">Manage Users</h2>
          <div className="overflow-x-auto">
            <table className="min-w-full bg-white border border-gray-200">
              <thead>
                <tr className="bg-gray-200">
                  <th className="py-2 px-4 border-b text-left text-gray-600">ID</th>
                  <th className="py-2 px-4 border-b text-left text-gray-600">Name</th>
                  <th className="py-2 px-4 border-b text-left text-gray-600">Email</th>
                  <th className="py-2 px-4 border-b text-left text-gray-600">Admin</th>
                  <th className="py-2 px-4 border-b text-left text-gray-600">Actions</th>
                </tr>
              </thead>
              <tbody>
                {users.map(user => (
                  <tr key={user.id} className="hover:bg-gray-50">
                    <td className="py-2 px-4 border-b">{user.id}</td>
                    <td className="py-2 px-4 border-b">{user.name}</td>
                    <td className="py-2 px-4 border-b">{user.email}</td>
                    <td className="py-2 px-4 border-b">{user.is_admin ? 'Yes' : 'No'}</td>
                    <td className="py-2 px-4 border-b">
                      <button
                        onClick={() => deleteUser(user.id)}
                        className="bg-red-500 hover:bg-red-600 text-white font-semibold py-1 px-3 rounded mr-2"
                      >
                        Delete
                      </button>
                      <button
                        onClick={() => toggleAdminStatus(user.id)}
                        className="bg-yellow-500 hover:bg-yellow-600 text-white font-semibold py-1 px-3 rounded"
                      >
                        {user.is_admin ? 'Remove Admin' : 'Make Admin'}
                      </button>
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    );
  };

  export default AdminPanel;