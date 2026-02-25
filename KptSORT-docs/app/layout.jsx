import { Footer, Link, Layout, Navbar } from 'nextra-theme-docs'
import { Banner, Head } from 'nextra/components'
import { getPageMap } from 'nextra/page-map'
import 'nextra-theme-docs/style.css'


export const metadata = {
}

const navbar = (
  <Navbar
    logo={<b>KptSORT-docs</b>}
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
        {/* Your additional tags should be passed as `children` of `<Head>` element */}
      </Head>
      <body>
        <Layout

          //banner={banner}
          navbar={navbar}
          pageMap={await getPageMap()}
          docsRepositoryBase="http://git.in.minelab/nagai/kpsort/-/tree/main/KptSORT-docs"
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