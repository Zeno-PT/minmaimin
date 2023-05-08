import 'date-fns';
import DateFnsUtils from '@date-io/date-fns';
import thLocale from "date-fns/locale/th";
import { Button, TextField } from '@material-ui/core';
import React, { useState } from 'react';
import { createStyles, makeStyles, Theme } from '@material-ui/core/styles';
import Grid from '@material-ui/core/Grid';
import { MuiPickersUtilsProvider, KeyboardDatePicker } from "@material-ui/pickers";
import Radio from '@material-ui/core/Radio';
import RadioGroup from '@material-ui/core/RadioGroup';
import FormControlLabel from '@material-ui/core/FormControlLabel';
import FormControl from '@material-ui/core/FormControl';
import InfoIcon from '@material-ui/icons/Info';
import Tooltip from '@material-ui/core/Tooltip';
import IconButton from '@material-ui/core/IconButton';
import Snackbar from '@material-ui/core/Snackbar';
import Paper from '@material-ui/core/Paper';
import Modal from '@material-ui/core/Modal';

const situationArray = ['inperson', 'public', 'chat', 'social', 'videocall'];
const msgTypeArray = ['verdict', 'court', 'none'];

function getModalStyle() {
    const top = 50;
    const left = 50;

    return {
        top: `${top}%`,
        left: `${left}%`,
        transform: `translate(-${top}%, -${left}%)`,
    };
}


const useStyles = makeStyles((theme: Theme) =>
    createStyles({
        root: {
            width: '83%',
            alignItems: 'center',
            margin: 'auto auto'
        },
        textField: {
            margin: 'auto auto'
        },
        submitButton: {
            backgroundColor: '#0561ff'
        },
        dropdown: {
            paddingLeft: '5px',
            height: '100%',
        },
        formControl: {
            margin: theme.spacing(3),
        },
        rowRadio: {
            justifyContent: 'space-around',
        },
        paper: {
            padding: '10px 20px',
            borderRadius: 20,
        },
        modalPaper: {
            position: 'absolute',
            width: 400,
            backgroundColor: theme.palette.background.paper,
            border: '2px solid #000',
            boxShadow: theme.shadows[5],
            padding: theme.spacing(2, 4, 3),
        }
    })
);

export default function SearchArea(props) {
    const classes = useStyles();
    const [date, setDate] = useState<Date | null>(new Date())
    const [text, setText] = useState<string | null>('');
    const [victim, setVictim] = useState<boolean | null>(null);
    const [thirdPerson, setThirdPerson] = useState<boolean | null>(null);
    const [situation, setSituation] = useState<number | null>(null);
    const [reaction, setReaction] = useState<boolean | null>(null);
    const [future, setFuture] = useState<boolean | null>(null);
    const [question, setQuestion] = useState<boolean | null>(null);
    const [msgType, setMsgType] = useState<number | null>(null);
    const [snackOpen, setSnackOpen] = useState(false);
    const [modalOpen, setModalOpen] = useState(false);
    const [modalStyle] = useState(getModalStyle);

    const modalBody = (
        <div style={modalStyle} className={classes.modalPaper}>
            <h2 id="simple-modal-title">เลือกสถานการณ์</h2>
            <p id="simple-modal-description">
                Duis mollis, est non commodo luctus, nisi erat porttitor ligula.
          </p>
        </div>
    );

    const handleModalOpen = () => {
        setModalOpen(true);
    };

    const handleModalClose = () => {
        setModalOpen(false);
    };

    function handleDateChange(date: Date | null) {
        setDate(date);
    };

    function handleTextChange(e: string) {
        setText(e);
    }

    function handleVictimChange(e: boolean) {
        setVictim(e);
    }

    function handleThirdPersonChange(e: boolean) {
        setThirdPerson(e);
    }

    function handleSituationChange(e: string) {
        setSituation(parseInt(e));
    }

    function handleReactionChange(e: boolean) {
        setReaction(e);
    }

    function handleFutureChange(e: boolean) {
        setFuture(e);
    }

    function handleQuestionChange(e: boolean) {
        setQuestion(e);
    }

    function handleMsgTypeChange(e: string) {
        setMsgType(parseInt(e));
    }

    function handleClearData() {
        setDate(new Date())
        setText('')
        setVictim(null)
        setThirdPerson(null)
        setSituation(null)
        setReaction(null)
        setFuture(null)
        setQuestion(null)
        setMsgType(null);
        setSnackOpen(true);
    }

    function handleSnackClose(event: React.SyntheticEvent | React.MouseEvent, reason?: string) {
        if (reason === 'clickaway') {
            return;
        }
        setSnackOpen(false);
    };

    function handleSubmit() {
        props.props.props.handleSubmit();
        props.props.props.handleDateCalculate(date);
    }

    function str2bool(value: string) {
        if (value && typeof value === "string") {
            if (value.toLowerCase() === "true") return true;
            if (value.toLowerCase() === "false") return false;
        }
        return false;
    }

    return (
        <div className={classes.root}>
            <form onSubmit={handleSubmit}>
                <FormControl component="fieldset" className={classes.formControl}>
                    <Grid container spacing={3} justify='flex-start'>
                        <Grid item xs={12}>
                            <Paper elevation={2} style={{ borderRadius: 5 }}>
                                <TextField
                                    id="outlined-multiline-static"
                                    label="ใส่ข้อความที่ต้องการประมวลผลที่นี่"
                                    multiline
                                    fullWidth={true}
                                    rows={8}
                                    defaultValue=""
                                    value={text}
                                    variant="outlined"
                                    className={classes.textField}
                                    onChange={(e) => { handleTextChange(e.target.value); }}
                                />
                            </Paper>
                        </Grid>
                        
                        <Grid item xs={3} />
                        <Grid item xs={6} style={{paddingTop: '0px', marginBottom: '10px'}}>
                            <h3>{"หรืออัพโหลดรูปภาพที่มีข้อความ"}</h3>
                            <Button
                                variant="contained"
                                component="label"
                                style={{padding: '7px 20px'}}
                            >
                                เลือกรูปภาพ
                                <input
                                    type="file"
                                    hidden
                                />
                            </Button>
                        </Grid>
                        <Grid item xs={3} />

                        <Grid item xs={1} />
                        <Grid item xs={10} className={classes.dropdown}>
                            <Paper elevation={2} className={classes.paper}>
                                โดนด่าอย่างไร?
                            <Tooltip title='สถานการณ์ขณะที่ถูกด่า'>
                                    <IconButton aria-label="info">
                                        <InfoIcon fontSize='small' />
                                    </IconButton>
                                </Tooltip>
                                <RadioGroup row className={classes.rowRadio} aria-label="สถานการณ์" name="สถานการณ์" value={situation} onChange={(e) => handleSituationChange(e.target.value)}>
                                    <FormControlLabel value={0} control={<Radio />} label="ต่อหน้า" />
                                    <FormControlLabel value={1} control={<Radio />} label="ที่ชุมชน" />
                                    <FormControlLabel value={2} control={<Radio />} label="ในแชทส่วนตัว" />
                                    <FormControlLabel value={3} control={<Radio />} label="โพสต์บนโซเชียลมีเดีย" />
                                    <FormControlLabel value={4} control={<Radio />} label="โทร / วิดีโอคอล" />
                                </RadioGroup>
                            </Paper>
                        </Grid>
                        <Grid item xs={1} />

                        <Grid item xs={1} />
                        <Grid item xs={10}>
                            <Grid container spacing={2}>
                                <Grid item xs={4} className={classes.dropdown}>
                                    <Paper elevation={2} className={classes.paper}>
                                        ยืนยันผู้เสียหายได้?
                                <Tooltip title='สามารถระบุได้ว่าเราถูกด่า เช่น ระบุชื่อ, หรือโพสต์รูปเรา'>
                                            <IconButton aria-label="info">
                                                <InfoIcon fontSize='small' />
                                            </IconButton>
                                        </Tooltip>
                                        <RadioGroup aria-label="ผู้เสียหาย" name="ผู้เสียหาย" value={victim} onChange={(e) => handleVictimChange(str2bool(e.target.value))}>
                                            <FormControlLabel value={true} control={<Radio />} label="ยืนยันได้" />
                                            <FormControlLabel value={false} control={<Radio />} label="ไม่สามารถยืนยันได้" />
                                        </RadioGroup>
                                    </Paper>
                                </Grid>
                                <Grid item xs={4} className={classes.dropdown}>
                                    <Paper elevation={2} className={classes.paper}>
                                        มีการด่าลับหลังหรือไม่?
                                <Tooltip title='เช่น เล่าให้คนอื่นฟัง, โพสต์ลงโซเชียล'>
                                            <IconButton aria-label="info">
                                                <InfoIcon fontSize='small' />
                                            </IconButton>
                                        </Tooltip>
                                        <RadioGroup aria-label="บุคคลที่สาม" name="บุคคลที่สาม" value={thirdPerson} onChange={(e) => handleThirdPersonChange(str2bool(e.target.value))}>
                                            <FormControlLabel value={true} control={<Radio />} label="มี" />
                                            <FormControlLabel value={false} control={<Radio />} label="ไม่มี" />
                                        </RadioGroup>
                                    </Paper>
                                </Grid>
                                <Grid item xs={4} className={classes.dropdown}>
                                    <Paper elevation={2} className={classes.paper}>
                                        มีการโต้ตอบกลับหรือไม่?
                                <Tooltip title='ด่ากันไปมา หรือมีการโต้ตอบระหว่างกันจะเข้าข้อยกเว้นที่ทำให้ไม่ผิดหมิ่นประมาท'>
                                            <IconButton aria-label="info">
                                                <InfoIcon fontSize='small' />
                                            </IconButton>
                                        </Tooltip>
                                        <RadioGroup aria-label="ตอบโต้" name="ตอบโต้" value={reaction} onChange={(e) => handleReactionChange(str2bool(e.target.value))}>
                                            <FormControlLabel value={true} control={<Radio />} label="มีการตอบโต้กลับ" />
                                            <FormControlLabel value={false} control={<Radio />} label="ไม่มีการตอบโต้" />
                                        </RadioGroup>
                                    </Paper>
                                </Grid>
                            </Grid>
                        </Grid>
                        <Grid item xs={1} />

                        <Grid item xs={1} />
                        <Grid item xs={10}>
                            <Grid container spacing={2} alignItems='stretch'>
                                <Grid item xs={4} className={classes.dropdown}>
                                    <Paper elevation={2} className={classes.paper}>
                                        พูดถึงเหตุการณ์ในอนาคตหรือไม่?
                                    <Tooltip title='เหตุการณ์ในอนาคต คือเหตุการณ์ที่ยังมาไม่ถึง เช่น ชาติหน้า อีก 10 ปีข้างหน้า'>
                                            <IconButton aria-label="info">
                                                <InfoIcon fontSize='small' />
                                            </IconButton>
                                        </Tooltip>
                                        <RadioGroup aria-label="อนาคต" name="อนาคต" value={future} onChange={(e) => handleFutureChange(str2bool(e.target.value))}>
                                            <FormControlLabel value={true} control={<Radio />} label="เป็น" />
                                            <FormControlLabel value={false} control={<Radio />} label="ไม่เป็น" />
                                        </RadioGroup>
                                    </Paper>
                                </Grid>
                                <Grid item xs={4} className={classes.dropdown}>
                                    <Paper elevation={2} className={classes.paper}>
                                        เป็นประโยคคำถามหรือไม่?
                                        <Tooltip title=''>
                                            <IconButton aria-label="info">
                                                <InfoIcon fontSize='small' />
                                            </IconButton>
                                        </Tooltip>
                                    <RadioGroup aria-label="คำถาม" name="คำถาม" value={question} onChange={(e) => handleQuestionChange(str2bool(e.target.value))}>
                                            <FormControlLabel value={true} control={<Radio />} label="เป็น" />
                                            <FormControlLabel value={false} control={<Radio />} label="ไม่เป็น" />
                                        </RadioGroup>
                                    </Paper>
                                </Grid>
                                <Grid item xs={4} className={classes.dropdown}>
                                    <Paper elevation={2} className={classes.paper}>
                                        ข้อความดังกล่าวมีลักษณะใด?
                                        <Tooltip title='วิพากษ์วิจารณ์ เช่น วิจารณ์การเมือง, ร้านอาหาร, สินค้า ฯลฯ'>
                                            <IconButton aria-label="info">
                                                <InfoIcon fontSize='small' />
                                            </IconButton>
                                        </Tooltip>
                                        <RadioGroup aria-label="ลักษณะ" name="ลักษณะ" value={msgType} onChange={(e) => handleMsgTypeChange(e.target.value)}>
                                            <FormControlLabel value={0} control={<Radio />} label="เป็นการวิพากษ์วิจารณ์" />
                                            <FormControlLabel value={1} control={<Radio />} label="เรื่องที่กล่าวเป็นคดีความในศาล" />
                                            <FormControlLabel value={2} control={<Radio />} label="ไม่มี" />
                                        </RadioGroup>
                                    </Paper>
                                </Grid>
                            </Grid>
                            </Grid>
                            <Grid item xs={1} />

                            <Grid item xs={1} />
                            <Grid item xs={10}>
                                <Paper elevation={2} className={classes.paper}>
                                    <p style={{ marginBottom: '0px', marginTop: '0px' }}>
                                        วันที่เกิดเหตุการณ์
                                <Tooltip title='ใช้เพื่อคำนวณว่าคดีขาดอายุความหรือยัง'>
                                            <IconButton aria-label="info">
                                                <InfoIcon fontSize='small' />
                                            </IconButton>
                                        </Tooltip>
                                    </p>
                                    <MuiPickersUtilsProvider utils={DateFnsUtils} locale={thLocale}>
                                        <KeyboardDatePicker
                                            style={{ width: '20%' }}
                                            margin='normal'
                                            id="date-picker-dialog"
                                            label=''
                                            format='dd/MM/yyyy'
                                            value={date}
                                            onChange={dateData => handleDateChange(dateData)}
                                            KeyboardButtonProps={{
                                                'aria-label': 'change date',
                                            }}
                                        />
                                    </MuiPickersUtilsProvider>
                                </Paper>
                            </Grid>
                            <Grid item xs={1} />

                        <Grid item xs={3} />
                        <Grid item xs={3} style={{ margin: '15px auto' }}>
                            <Button variant='contained' color='primary' size='large' className={classes.submitButton} onClick={handleSubmit}>
                                ประเมิน
                            </Button>
                        </Grid>
                        <Grid item xs={3} style={{ margin: '15px auto' }}>
                            <Button variant='contained' size='large' onClick={handleClearData}>
                                ล้างข้อมูล
                            </Button>
                        </Grid>
                        <Grid item xs={3} />
                    </Grid>
                </FormControl>
            </form>
            <Snackbar
                anchorOrigin={{
                    vertical: 'bottom',
                    horizontal: 'left',
                }}
                open={snackOpen}
                autoHideDuration={6000}
                onClose={handleSnackClose}
                message="ล้างข้อมูลเสร็จสิ้น"
                action={
                    <React.Fragment>
                        <Button color="secondary" size="small" onClick={handleSnackClose}>
                            ปิด
                    </Button>
                    </React.Fragment>
                }
            />
        </div>
    )
}