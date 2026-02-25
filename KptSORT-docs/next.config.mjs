import nextra from 'nextra'

const nextConfig = {
  output: 'export',
  basePath: '/KptSORT-docs',
  trailingSlash: true,
  images: {
    unoptimized: true
  }
}

const withNextra = nextra({
})

export default withNextra(nextConfig)