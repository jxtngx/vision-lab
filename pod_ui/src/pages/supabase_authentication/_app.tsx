import { Auth } from '@supabase/ui'
import { supabase } from '../../../lib/initSupabase'

export default function MyApp({ Component: any, pageProps: any }) {
  return (
    <main className={'dark'}>
      <Auth.UserContextProvider supabaseClient={supabase}>
        <Component {...pageProps} />
      </Auth.UserContextProvider>
    </main>
  )
}
