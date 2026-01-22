import 'package:flutter/material.dart';
import 'package:provider/provider.dart';

import '../../providers/auth_provider.dart';
import '../../providers/job_provider.dart';
import '../../theme/app_theme.dart';
import '../../widgets/form_input.dart';
import '../../widgets/primary_button.dart';
import '../recruiter/recruiter_main_page.dart';
import '../onboarding_page.dart';
import '../seeker/seeker_shell.dart';
import 'register_page.dart';

class LoginPage extends StatefulWidget {
  const LoginPage({super.key, this.initialRole = 'job_seeker'});
  final String initialRole;

  @override
  State<LoginPage> createState() => _LoginPageState();
}

class _LoginPageState extends State<LoginPage> {
  final emailCtrl = TextEditingController();
  final passwordCtrl = TextEditingController();
  late String selectedRole;
  String? emailError;
  String? passwordError;

  @override
  void initState() {
    super.initState();
    selectedRole = widget.initialRole;
  }

  @override
  void dispose() {
    emailCtrl.dispose();
    passwordCtrl.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final auth = context.watch<AuthProvider>();
    return Scaffold(
      appBar: AppBar(
        leading: IconButton(
          icon: const Icon(Icons.arrow_back_ios_new_rounded),
          onPressed: () => Navigator.pushReplacement(context,
              MaterialPageRoute(builder: (_) => const OnboardingPage())),
        ),
        title: const Text('Welcome back'),
      ),
      body: SafeArea(
        child: SingleChildScrollView(
          padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const SizedBox(height: 12),
              const SizedBox(height: 12),
              FormInput(
                  label: 'Email',
                  controller: emailCtrl,
                  hint: 'alex@example.com',
                  keyboardType: TextInputType.emailAddress),
              if (emailError != null) _inlineError(emailError!),
              const SizedBox(height: 16),
              FormInput(
                  label: 'Password',
                  controller: passwordCtrl,
                  hint: '••••••••',
                  obscure: true),
              if (passwordError != null) _inlineError(passwordError!),
              const SizedBox(height: 20),
              PrimaryButton(
                label: 'Sign In',
                loading: auth.isLoading,
                onPressed: () async {
                  final email = emailCtrl.text.trim();
                  final password = passwordCtrl.text;
                  setState(() {
                    emailError = _requiredError(email);
                    passwordError = _requiredError(password);
                  });
                  await auth.login(email, password, selectedRole);
                  if (!mounted) return;
                  if (auth.session != null) {
                    _goToDashboard(context, auth);
                    context.read<JobProvider>().loadJobs();
                  } else if (auth.error != null) {
                    _applyServerError(auth.error!);
                  }
                },
              ),
              const SizedBox(height: 20),
              Row(
                mainAxisAlignment: MainAxisAlignment.center,
                children: [
                  const Text('No account yet? '),
                  TextButton(
                    onPressed: () {
                      Navigator.push(
                          context,
                          MaterialPageRoute(
                              builder: (_) =>
                                  RegisterPage(initialRole: selectedRole)));
                    },
                    child: const Text('Create one',
                        style: TextStyle(
                            color: AppTheme.primary,
                            fontWeight: FontWeight.w700)),
                  ),
                ],
              )
            ],
          ),
        ),
      ),
    );
  }

  void _goToDashboard(BuildContext context, AuthProvider auth) {
    final role = auth.session?.profile.role;
    if (role == 'recruiter') {
      Navigator.pushAndRemoveUntil(
          context,
          MaterialPageRoute(builder: (_) => const RecruiterMainPage()),
          (_) => false);
    } else {
      Navigator.pushAndRemoveUntil(context,
          MaterialPageRoute(builder: (_) => const SeekerShell()), (_) => false);
    }
  }

  void _showError(BuildContext context, String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(
        content: Text(message, maxLines: 2, overflow: TextOverflow.ellipsis),
        behavior: SnackBarBehavior.floating,
      ),
    );
  }

  String? _requiredError(String value) {
    if (value.trim().isEmpty) {
      return _friendlyMessage('required');
    }
    return null;
  }

  String _friendlyMessage(String message) {
    final lower = message.toLowerCase();
    if (lower.contains('too short')) {
      return 'Password must be at least 6 characters.';
    }
    if (lower.contains('invalid email')) {
      return 'Please enter a valid email address.';
    }
    if (lower.contains('required')) {
      return 'Please fill in this field.';
    }
    if (lower.contains('account not found')) {
      return 'Account does not exist for this role. Switch to the correct login page.';
    }
    return message;
  }

  void _applyServerError(String message) {
    final friendly = _friendlyMessage(message);
    final lower = message.toLowerCase();
    setState(() {
      if (lower.contains('invalid email') || lower.contains('email')) {
        emailError = friendly;
        return;
      }
      if (lower.contains('too short') || lower.contains('password')) {
        passwordError = friendly;
        return;
      }
      if (lower.contains('required')) {
        if (emailCtrl.text.trim().isEmpty) {
          emailError = friendly;
        }
        if (passwordCtrl.text.isEmpty) {
          passwordError = friendly;
        }
      }
    });
    _showError(context, friendly);
  }

  Widget _inlineError(String message) {
    return Padding(
      padding: const EdgeInsets.only(left: 4, top: 6),
      child: Text(
        message,
        maxLines: 2,
        overflow: TextOverflow.ellipsis,
        style: const TextStyle(
            color: Colors.red, fontWeight: FontWeight.w600, fontSize: 12),
      ),
    );
  }
}
