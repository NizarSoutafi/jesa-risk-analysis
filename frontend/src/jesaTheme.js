// src/jesaTheme.js
import { createTheme } from '@mui/material/styles';

const JESA_BLUE = '#1d3b7e';
const JESA_WHITE = '#ffffff';

const jesaTheme = createTheme({
  palette: {
    primary: {
      main: JESA_BLUE,
      contrastText: JESA_WHITE,
    },
    secondary: { // For elements that need to contrast, often on primary background
      main: JESA_WHITE,
      contrastText: JESA_BLUE,
    },
    background: {
      default: '#f8f9fa', // A very light, almost white, neutral gray for page backgrounds
      paper: JESA_WHITE,
    },
    text: {
      primary: '#212529',   // A very dark gray, near black, for high contrast on white
      secondary: '#495057', // A softer gray for secondary text
      disabled: '#adb5bd',  // For disabled text
    },
    action: {
      active: JESA_BLUE,
      hover: 'rgba(29, 59, 126, 0.06)', // Subtle blue hover for light backgrounds
      selected: 'rgba(29, 59, 126, 0.1)',
      disabled: 'rgba(0, 0, 0, 0.26)',
      disabledBackground: 'rgba(0, 0, 0, 0.12)',
    },
    divider: 'rgba(29, 59, 126, 0.12)', // Subtle blue-tinted divider
  },
  typography: {
    fontFamily: "'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Oxygen, Ubuntu, Cantarell, 'Open Sans', 'Helvetica Neue', sans-serif",
    h1: { fontSize: '2.5rem', fontWeight: 600, color: JESA_BLUE, lineHeight: 1.2 },
    h2: { fontSize: '2rem', fontWeight: 600, color: JESA_BLUE, lineHeight: 1.2 },
    h3: { fontSize: '1.75rem', fontWeight: 600, color: JESA_BLUE, lineHeight: 1.3 }, // Main page/section titles
    h4: { fontSize: '1.5rem', fontWeight: 600, color: JESA_BLUE, lineHeight: 1.3 },   // Card titles
    h5: { fontSize: '1.25rem', fontWeight: 500, color: JESA_BLUE, lineHeight: 1.4 },  // Sub-card titles
    h6: { fontSize: '1.1rem', fontWeight: 500, color: JESA_BLUE, lineHeight: 1.4 },   // Important text/labels
    subtitle1: { fontSize: '1rem', fontWeight: 500, color: JESA_WHITE }, // Sidebar section titles
    subtitle2: { fontSize: '0.875rem', fontWeight: 400, color: 'text.secondary' },
    body1: { fontSize: '1rem', fontWeight: 400, lineHeight: 1.6 },
    body2: { fontSize: '0.875rem', fontWeight: 400, lineHeight: 1.5 },
    caption: { fontSize: '0.75rem', fontWeight: 400, color: 'text.secondary' },
    overline: { fontSize: '0.75rem', fontWeight: 600, textTransform: 'uppercase', letterSpacing: '1px' },
    button: {
      textTransform: 'none', // Less shouty buttons
      fontWeight: 500,
    }
  },
  spacing: 8, // Base spacing unit (Material-UI default)
  shape: {
    borderRadius: 8, // Consistent border radius
  },
  components: {
    MuiDrawer: {
      styleOverrides: {
        paper: {
          backgroundColor: JESA_BLUE,
          color: JESA_WHITE,
          borderRight: 'none', // Remove any default border
        },
      },
    },
    MuiButton: {
      defaultProps: {
        disableElevation: true, // Flatter buttons by default
      },
      styleOverrides: {
        root: {
          borderRadius: '6px', // Slightly less than global for buttons
          padding: '8px 16px',
        },
        containedPrimary: {
          '&:hover': {
            backgroundColor: '#17306b', // Darker JESA blue
          },
        },
        containedSecondary: {
          '&:hover': {
            backgroundColor: 'rgba(29, 59, 126, 0.08)', // Subtle blue tint on white hover
          },
        },
        outlinedPrimary: {
            borderColor: JESA_BLUE,
            color: JESA_BLUE,
            '&:hover': {
                backgroundColor: 'rgba(29, 59, 126, 0.04)', // Lightest blue tint
                borderColor: '#17306b',
            }
        },
        textPrimary: { // For text buttons that should be blue
            color: JESA_BLUE,
            '&:hover': {
                backgroundColor: 'rgba(29, 59, 126, 0.04)',
            }
        }
      },
    },
    MuiListItemButton: {
      styleOverrides: {
        root: {
          color: JESA_WHITE,
          borderRadius: '6px',
          margin: '4px 0',
          '&:hover': {
            backgroundColor: 'rgba(255, 255, 255, 0.1)',
          },
          '&.Mui-selected': {
            backgroundColor: 'rgba(255, 255, 255, 0.15)',
            '&:hover': {
              backgroundColor: 'rgba(255, 255, 255, 0.2)',
            },
          },
          '& .MuiListItemText-primary': {
            color: JESA_WHITE,
            fontWeight: 400,
          },
          '& .MuiSvgIcon-root': { // Icon color in list item
            color: 'rgba(255, 255, 255, 0.8)',
          }
        },
      },
    },
    MuiCard: {
      styleOverrides: {
        root: {
          boxShadow: '0px 5px 15px rgba(29, 59, 126, 0.08)', // Softer, blue-tinted shadow
          // border: '1px solid rgba(29, 59, 126, 0.12)', // Optional subtle border
        }
      }
    },
    MuiAppBar: { // If you add an AppBar later
        styleOverrides: {
            colorPrimary: {
                backgroundColor: JESA_WHITE,
                color: JESA_BLUE,
                boxShadow: '0px 2px 4px -1px rgba(29, 59, 126, 0.1)',
            }
        }
    },
    MuiTextField: { // General styling for TextFields (on light backgrounds)
        defaultProps: {
            variant: 'outlined',
            size: 'small',
        },
    },
    // Styles for Form elements within the JESA_BLUE Drawer
    MuiInputLabel: {
      styleOverrides: {
        root: {
          '&.MuiInputLabel-root:not(.Mui-focused):not(.MuiFormLabel-filled)[data-shrink="false"]': { // Targeting placeholder state within Drawer
            '.MuiDrawer-root &': { color: 'rgba(255, 255, 255, 0.6)' },
          },
          '.MuiDrawer-root &': { color: 'rgba(255, 255, 255, 0.8)' }, // Default label color in drawer
          '&.Mui-focused.MuiInputLabel-root': {
            '.MuiDrawer-root &': { color: JESA_WHITE }, // Focused label color in drawer
          }
        },
      },
    },
    MuiOutlinedInput: {
      styleOverrides: {
        root: {
          '.MuiDrawer-root &': { // For inputs inside the Drawer
            color: JESA_WHITE,
            '& .MuiOutlinedInput-notchedOutline': {
              borderColor: 'rgba(255, 255, 255, 0.3)',
            },
            '&:hover .MuiOutlinedInput-notchedOutline': {
              borderColor: 'rgba(255, 255, 255, 0.6)',
            },
            '&.Mui-focused .MuiOutlinedInput-notchedOutline': {
              borderColor: JESA_WHITE,
              borderWidth: '1px',
            },
            '& .MuiSvgIcon-root': {
                color: 'rgba(255, 255, 255, 0.6)',
            }
          },
        },
      },
    },
    MuiSlider: { // For the probability slider in the Drawer
        styleOverrides: {
            root: {
                '.MuiDrawer-root &': {
                    color: JESA_WHITE,
                }
            },
            thumb: {
                '.MuiDrawer-root &': {
                    backgroundColor: JESA_WHITE,
                }
            },
            rail: {
                '.MuiDrawer-root &': {
                    opacity: 0.4,
                    backgroundColor: JESA_WHITE,
                }
            },
            track: {
                '.MuiDrawer-root &': {
                    backgroundColor: JESA_WHITE,
                }
            }
        }
    },
    MuiTableHead: {
        styleOverrides: {
            root: {
                backgroundColor: 'rgba(29, 59, 126, 0.05)', // Very light blue for table head
                '& .MuiTableCell-head': {
                    color: JESA_BLUE,
                    fontWeight: 600,
                }
            }
        }
    },
    MuiTableRow: {
        styleOverrides: {
            root: {
                '&:hover': {
                    backgroundColor: 'rgba(29, 59, 126, 0.03)', // Subtle hover for table rows
                }
            }
        }
    }
  },
});

export default jesaTheme;