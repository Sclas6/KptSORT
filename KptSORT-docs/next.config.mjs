import nextra from 'nextra'

const nextConfig = {
  output: 'export',
  basePath: '/KptSORT',
  trailingSlash: true,
  images: {
    unoptimized: true
  }
}

const withNextra = nextra({
})

export default withNextra(nextConfig)