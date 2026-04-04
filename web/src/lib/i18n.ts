import { useState, useEffect } from 'react';
import { getStatus } from './api';
import type { TranslationKey } from './i18n-keys';
export type { TranslationKey };

// ---------------------------------------------------------------------------
// Load per-locale JSON files at build time via Vite glob import
// ---------------------------------------------------------------------------

const localeModules = import.meta.glob<Record<string, string>>('./locales/*.json', { eager: true });

export type Locale = 'ar' | 'bn' | 'cs' | 'da' | 'de' | 'el' | 'en' | 'es' | 'fi' | 'fr' | 'he' | 'hi' | 'hu' | 'id' | 'it' | 'ja' | 'ko' | 'nb' | 'nl' | 'pl' | 'pt' | 'ro' | 'ru' | 'sv' | 'th' | 'tl' | 'tr' | 'uk' | 'ur' | 'vi' | 'zh';

const translations: Record<Locale, Record<string, string>> = {} as Record<Locale, Record<string, string>>;

for (const [path, mod] of Object.entries(localeModules)) {
  // path is like './locales/en.json' — extract the locale code
  const match = path.match(/\.\/locales\/(.+)\.json$/);
  if (match) {
    const locale = match[1] as Locale;
    // Vite eager JSON imports expose the object as `default` or directly
    translations[locale] = (mod as { default?: Record<string, string> }).default ?? mod;
  }
}

// ---------------------------------------------------------------------------
// Current locale state
// ---------------------------------------------------------------------------

let currentLocale: Locale = 'en';

export function getLocale(): Locale {
  return currentLocale;
}

export function setLocale(locale: Locale): void {
  currentLocale = locale;
}

// ---------------------------------------------------------------------------
// Translation function
// ---------------------------------------------------------------------------

/**
 * Translate a key using the current locale. Returns the key itself if no
 * translation is found.
 */
export function t(key: TranslationKey): string {
  return translations[currentLocale]?.[key] ?? translations.en[key] ?? key;
}

/**
 * Get the translation for a specific locale. Falls back to English, then to the
 * raw key.
 */
export function tLocale(key: TranslationKey, locale: Locale): string {
  return translations[locale]?.[key] ?? translations.en[key] ?? key;
}

// ---------------------------------------------------------------------------
// Supported locales
// ---------------------------------------------------------------------------

export const SUPPORTED_LOCALES: { code: Locale; name: string; flag: string }[] = [
  { code: 'ar', name: '\u0627\u0644\u0639\u0631\u0628\u064a\u0629', flag: '\ud83c\uddf8\ud83c\udde6' },
  { code: 'bn', name: '\u09ac\u09be\u0982\u09b2\u09be', flag: '\ud83c\udde7\ud83c\udde9' },
  { code: 'cs', name: '\u010ce\u0161tina', flag: '\ud83c\udde8\ud83c\uddff' },
  { code: 'da', name: 'Dansk', flag: '\ud83c\udde9\ud83c\uddf0' },
  { code: 'de', name: 'Deutsch', flag: '\ud83c\udde9\ud83c\uddea' },
  { code: 'el', name: '\u0395\u03bb\u03bb\u03b7\u03bd\u03b9\u03ba\u03ac', flag: '\ud83c\uddec\ud83c\uddf7' },
  { code: 'en', name: 'English', flag: '\ud83c\uddfa\ud83c\uddf8' },
  { code: 'es', name: 'Espa\u00f1ol', flag: '\ud83c\uddea\ud83c\uddf8' },
  { code: 'fi', name: 'Suomi', flag: '\ud83c\uddeb\ud83c\uddee' },
  { code: 'fr', name: 'Fran\u00e7ais', flag: '\ud83c\uddeb\ud83c\uddf7' },
  { code: 'he', name: '\u05e2\u05d1\u05e8\u05d9\u05ea', flag: '\ud83c\uddee\ud83c\uddf1' },
  { code: 'hi', name: '\u0939\u093f\u0928\u094d\u0926\u0940', flag: '\ud83c\uddee\ud83c\uddf3' },
  { code: 'hu', name: 'Magyar', flag: '\ud83c\udded\ud83c\uddfa' },
  { code: 'id', name: 'Bahasa Indonesia', flag: '\ud83c\uddee\ud83c\udde9' },
  { code: 'it', name: 'Italiano', flag: '\ud83c\uddee\ud83c\uddf9' },
  { code: 'ja', name: '\u65e5\u672c\u8a9e', flag: '\ud83c\uddef\ud83c\uddf5' },
  { code: 'ko', name: '\ud55c\uad6d\uc5b4', flag: '\ud83c\uddf0\ud83c\uddf7' },
  { code: 'nb', name: 'Norsk', flag: '\ud83c\uddf3\ud83c\uddf4' },
  { code: 'nl', name: 'Nederlands', flag: '\ud83c\uddf3\ud83c\uddf1' },
  { code: 'pl', name: 'Polski', flag: '\ud83c\uddf5\ud83c\uddf1' },
  { code: 'pt', name: 'Portugu\u00eas', flag: '\ud83c\udde7\ud83c\uddf7' },
  { code: 'ro', name: 'Rom\u00e2n\u0103', flag: '\ud83c\uddf7\ud83c\uddf4' },
  { code: 'ru', name: '\u0420\u0443\u0441\u0441\u043a\u0438\u0439', flag: '\ud83c\uddf7\ud83c\uddfa' },
  { code: 'sv', name: 'Svenska', flag: '\ud83c\uddf8\ud83c\uddea' },
  { code: 'th', name: '\u0e44\u0e17\u0e22', flag: '\ud83c\uddf9\ud83c\udded' },
  { code: 'tl', name: 'Filipino', flag: '\ud83c\uddf5\ud83c\udded' },
  { code: 'tr', name: 'T\u00fcrk\u00e7e', flag: '\ud83c\uddf9\ud83c\uddf7' },
  { code: 'uk', name: '\u0423\u043a\u0440\u0430\u0457\u043d\u0441\u044c\u043a\u0430', flag: '\ud83c\uddfa\ud83c\udde6' },
  { code: 'ur', name: '\u0627\u0631\u062f\u0648', flag: '\ud83c\uddf5\ud83c\uddf0' },
  { code: 'vi', name: 'Ti\u1ebfng Vi\u1ec7t', flag: '\ud83c\uddfb\ud83c\uddf3' },
  { code: 'zh', name: '\u4e2d\u6587', flag: '\ud83c\udde8\ud83c\uddf3' },
];

// ---------------------------------------------------------------------------
// React hook
// ---------------------------------------------------------------------------

/**
 * React hook that fetches the locale from /api/status on mount and keeps the
 * i18n module in sync. Returns the current locale and a `t` helper bound to it.
 */
export function useLocale(): { locale: Locale; t: (key: TranslationKey) => string } {
  const [locale, setLocaleState] = useState<Locale>(currentLocale);

  useEffect(() => {
    let cancelled = false;

    getStatus()
      .then((status) => {
        if (cancelled) return;
        const raw = (status.locale || 'en').toLowerCase().replace(/-.*/, '').replace(/_.*/, '');
        const detected: Locale = (raw in translations) ? (raw as Locale) : 'en';
        setLocale(detected);
        setLocaleState(detected);
      })
      .catch(() => {
        // Keep default locale on error
      });

    return () => {
      cancelled = true;
    };
  }, []);

  return {
    locale,
    t: (key: TranslationKey) => tLocale(key, locale),
  };
}
