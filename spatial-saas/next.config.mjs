/** @type {import('next').NextConfig} */
const nextConfig = {
  output: 'standalone',
  eslint: {
    ignoreDuringBuilds: true,
  },
  typescript: {
    ignoreBuildErrors: true,
  },
  experimental: {},
  transpilePackages: ['three', '@react-three/fiber', '@react-three/drei'],
  async rewrites() {
    const backend = process.env.BACKEND_INTERNAL_URL || "http://127.0.0.1:8000";
    return [
      {
        source: '/health',
        destination: `${backend}/health`,
      },
      {
        source: '/tasks',
        destination: `${backend}/tasks`,
      },
      {
        source: '/actions',
        destination: `${backend}/actions`,
      },
      {
        source: '/reset',
        destination: `${backend}/reset`,
      },
      {
        source: '/step',
        destination: `${backend}/step`,
      },
      {
        source: '/state',
        destination: `${backend}/state`,
      },
      {
        source: '/leaderboard',
        destination: `${backend}/leaderboard`,
      },
      {
        source: '/episodes/:path*',
        destination: `${backend}/episodes/:path*`,
      },
      {
        source: '/fabricate',
        destination: `${backend}/fabricate`,
      },
      {
        source: '/detect-deepfake',
        destination: `${backend}/detect-deepfake`,
      },
      {
        source: '/detect-deepfake/status',
        destination: `${backend}/detect-deepfake/status`,
      },
    ];
  },
};

export default nextConfig;
