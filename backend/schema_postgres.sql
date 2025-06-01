-- Drop the existing table if it exists with the old structure to avoid conflicts
-- Be cautious with DROP TABLE if you have important data; backup first.
-- Alternatively, you might need an ALTER TABLE statement for existing databases.
DROP TABLE IF EXISTS users CASCADE; -- CASCADE to drop dependent objects if any
DROP TABLE IF EXISTS reset_codes CASCADE;

CREATE TABLE IF NOT EXISTS users (
    id SERIAL PRIMARY KEY,
    name TEXT NOT NULL,
    email TEXT UNIQUE NOT NULL,
    password TEXT NOT NULL,
    is_admin BOOLEAN NOT NULL DEFAULT FALSE -- Changed from 'role TEXT'
);

CREATE TABLE IF NOT EXISTS reset_codes (
    email TEXT PRIMARY KEY,
    code TEXT NOT NULL,
    expires_at TIMESTAMP WITH TIME ZONE NOT NULL, -- Use TIMESTAMP WITH TIME ZONE for better date handling
    FOREIGN KEY (email) REFERENCES users(email) ON DELETE CASCADE
);