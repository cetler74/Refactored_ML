import ccxt.async_support as ccxt
import asyncio
import re

async def test_prefixed_tokens():
    try:
        print("Testing prefixed tokens on Binance...")
        exchange = ccxt.binance()
        
        # Fetch all tickers
        tickers = await exchange.fetch_tickers()
        print(f"Total tickers found: {len(tickers)}")
        
        # Check for PEPE, SHIB, BONK tokens
        print("\nPEPE tokens:")
        pepe_tokens = [t for t in tickers.keys() if 'PEPE' in t]
        for token in pepe_tokens:
            print(f"  {token}")
        
        print("\nSHIB tokens:")
        shib_tokens = [t for t in tickers.keys() if 'SHIB' in t]
        for token in shib_tokens:
            print(f"  {token}")
        
        print("\nBONK tokens:")
        bonk_tokens = [t for t in tickers.keys() if 'BONK' in t]
        for token in bonk_tokens:
            print(f"  {token}")
        
        # Find tokens with numeric prefixes
        numeric_prefixed = []
        for token in tickers.keys():
            # Check if token starts with a number
            if re.match(r'^\d+', token) and token != '1INCH':
                numeric_prefixed.append(token)
        
        print(f"\nFound {len(numeric_prefixed)} tokens with numeric prefixes:")
        for token in sorted(numeric_prefixed)[:20]:  # Print first 20 for brevity
            print(f"  {token}")
        
        if len(numeric_prefixed) > 20:
            print(f"  ... and {len(numeric_prefixed) - 20} more")
        
        # Check USDC pairs specifically
        usdc_pairs = [t for t in tickers.keys() if t.endswith('USDC')]
        print(f"\nFound {len(usdc_pairs)} USDC pairs")
        
        # Check for prefixed USDC pairs
        prefixed_usdc = [t for t in usdc_pairs if re.match(r'^\d+', t) and t != '1INCHUSDC']
        print(f"\nFound {len(prefixed_usdc)} USDC pairs with numeric prefixes:")
        for token in sorted(prefixed_usdc):
            print(f"  {token}")
        
    except Exception as e:
        print(f"Error: {str(e)}")
    finally:
        await exchange.close()

asyncio.run(test_prefixed_tokens()) 