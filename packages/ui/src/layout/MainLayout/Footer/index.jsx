// ========== Footer ========== //
import { Box, Typography } from '@mui/material'

const Footer = () => {
    return (
        <Box
            sx={{
                width: '100%',
                textAlign: 'center',
                padding: '16px 0',
                backgroundColor: (theme) => theme.palette.background.default,
                borderTop: '1px solid',
                borderColor: (theme) => theme.palette.primary[200] + 75,
                position: 'relative' // Adjusted to ensure proper layout
            }}
        >
            <Typography variant='body2' color='textSecondary'>
                © 2024 Tribus Digital. All rights reserved.
            </Typography>
        </Box>
    )
}

export default Footer
