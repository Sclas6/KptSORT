import { Footer, Link, Layout, Navbar } from 'nextra-theme-docs'
import { Banner, Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'


export const metadata = {
}

const navbar = (
  <Navbar
    logo={<b>KptSORT</b>}
  //logoLink="http://git.in.minelab"
  // ... Your additional navbar options
  />
)
const footer = <Footer>MIT {new Date().getFullYear()} © Nextra.</Footer>

export default async function RootLayout({ children }) {
  return (
    <html
      // Not required, but good for SEO
      lang="en"
      // Required to be set
      dir="ltr"
      // Suggested by `next-themes` package https://github.com/pacocoursey/next-themes#with-app
      suppressHydrationWarning
    >
      <Head
      // ... Your additional head options
      >
        <link rel="icon" href="data:image/svg+xml,<svg xmlns=%22http://www.w3.org/2000/svg%22 viewBox=%220 0 100 100%22><text y=%22.9em%22 font-size=%2290%22>🐝</text></svg>"/>
        {/* Your additional tags should be passed as `children` of `<Head>` element */}
      </Head>
      <body>
        <Layout

          //banner={banner}
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase="http://git.in.minelab/nagai/kpsort/-/tree/main/KptSORT"
          footer={footer}
          feedback={{ "link": "http://git.in.minelab/nagai/kpsort/-/issues/new" }}
        // ... Your additional layout options
        >
          {children}
        </Layout>
      </body>
    </html>
  )
}