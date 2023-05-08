import React from 'react';
import { createStyles, makeStyles, Theme } from '@material-ui/core/styles';


const useStyles = makeStyles((theme: Theme) => 
    createStyles({
        root: {
            marginTop: '3em',
            marginBottom: '2em'
        }
    })
);

export default function SearchHeader() {
    const classes = useStyles();

    return (
        <div className={classes.root}>
            <h1>กรอกรายละเอียดประเมินโอกาสเข้าข่ายหมิ่นประมาท</h1>
        </div>
    )
}